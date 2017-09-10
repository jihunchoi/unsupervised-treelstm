import argparse

import numpy as np
import torch
from torchtext import data, datasets

from sst.model import SSTModel
from utils.helper import wrap_with_variable, unwrap_scalar_variable


def evaluate(args):
    text_field = data.Field(lower=args.lower, include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False)

    filter_pred = None
    if not args.fine_grained:
        filter_pred = lambda ex: ex.label != 'neutral'
    dataset_splits = datasets.SST.splits(
        root='./data/sst', text_field=text_field, label_field=label_field,
        fine_grained=args.fine_grained, train_subtrees=True,
        filter_pred=filter_pred)
    test_dataset = dataset_splits[2]

    text_field.build_vocab(*dataset_splits)
    label_field.build_vocab(*dataset_splits)

    print(f'Number of classes: {len(label_field.vocab)}')

    _, _, test_loader = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)

    num_classes = len(label_field.vocab)
    model = SSTModel(num_classes=num_classes, num_words=len(text_field.vocab),
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout)
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')
    model.load_state_dict(torch.load(args.model))
    model.eval()
    if args.gpu > -1:
        model.cuda(args.gpu)
    num_correct = 0
    num_data = len(test_dataset)
    for batch in test_loader:
        words, length = batch.text
        label = batch.label
        length = wrap_with_variable(length, volatile=True, gpu=args.gpu)
        logits = model(words=words, length=length)
        label_pred = logits.max(1)[1]
        num_correct_batch = torch.eq(label, label_pred).long().sum()
        num_correct_batch = unwrap_scalar_variable(num_correct_batch)
        num_correct += num_correct_batch
    print(f'# data: {num_data}')
    print(f'# correct: {num_correct}')
    print(f'Accuracy: {num_correct / num_data:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
