import argparse
import logging
import os
import pickle
from functools import partial

import tensorboard
from tensorboard import summary

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader

from sst.model import SSTModel
from sst.utils.dataset import SSTDataset
from utils.glove import load_glove
from utils.helper import wrap_with_variable, unwrap_scalar_variable


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def train(args):
    with open(args.train_data, 'rb') as f:
        train_dataset: SSTDataset = pickle.load(f)
    with open(args.valid_data, 'rb') as f:
        valid_dataset: SSTDataset = pickle.load(f)

    train_collate_fn = partial(train_dataset.collate, omit_prob=args.omit_prob)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              collate_fn=train_collate_fn,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2,
                              collate_fn=valid_dataset.collate,
                              pin_memory=True)
    word_vocab = train_dataset.word_vocab

    num_classes = train_dataset.num_classes
    model = SSTModel(num_classes=num_classes, num_words=len(word_vocab),
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout)
    if args.glove:
        logging.info('Loading GloVe pretrained vectors...')
        model.word_embedding.weight.data.zero_()
        glove_weight = load_glove(
            path=args.glove, vocab=word_vocab,
            init_weight=model.word_embedding.weight.data.numpy())
        glove_weight[word_vocab.pad_id] = 0
        model.word_embedding.weight.data.set_(torch.FloatTensor(glove_weight))
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    if args.gpu > -1:
        logging.info(f'Using GPU {args.gpu}')
        model.cuda(args.gpu)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5, patience=10000000,
        verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'train'), flush_secs=10)
    valid_summary_writer = tensorboard.FileWriter(
        logdir=os.path.join(args.save_dir, 'log', 'valid'), flush_secs=10)

    def run_iter(batch, is_training):
        model.train(is_training)
        words = wrap_with_variable(batch['words'], volatile=not is_training,
                                   gpu=args.gpu)
        length = wrap_with_variable(batch['length'], volatile=not is_training,
                                    gpu=args.gpu)
        label = wrap_with_variable(batch['label'], volatile=not is_training,
                                   gpu=args.gpu)
        logits = model(words=words, length=length)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        value = unwrap_scalar_variable(value)
        summ = summary.scalar(name=name, scalar=value)
        summary_writer.add_summary(summary=summ, global_step=step)

    num_train_batches = len(train_loader)
    validate_every = num_train_batches // 20
    best_vaild_accuacy = 0
    iter_count = 0
    for epoch_num in range(1, args.max_epoch + 1):
        logging.info(f'Epoch {epoch_num}: start')
        for batch_iter, train_batch in enumerate(train_loader):
            train_loss, train_accuracy = run_iter(
                batch=train_batch, is_training=True)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)

            if (batch_iter + 1) % validate_every == 0:
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy = run_iter(
                        batch=valid_batch, is_training=False)
                    valid_loss_sum += unwrap_scalar_variable(valid_loss)
                    valid_accuracy_sum += unwrap_scalar_variable(valid_accuracy)
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                scheduler.step(valid_accuracy)
                progress = epoch_num + batch_iter/num_train_batches
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid loss = {valid_loss:.4f}, '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                                      f'-{valid_loss:.4f}'
                                      f'-{valid_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--valid-data', required=True)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--glove', default=None)
    parser.add_argument('--fix-word-embedding', default=False,
                        action='store_true')
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--omit-prob', default=0.0, type=float)
    parser.add_argument('--optimizer', default='adam')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
