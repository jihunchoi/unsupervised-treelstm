import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from sst.model import SSTModel
from sst.utils.dataset import SSTDataset
from utils.helper import wrap_with_variable, unwrap_scalar_variable


def evaluate(args):
    with open(args.data, 'rb') as f:
        test_dataset: SSTDataset = pickle.load(f)
    word_vocab = test_dataset.word_vocab
    num_classes = test_dataset.num_classes
    model = SSTModel(num_classes=num_classes, num_words=len(word_vocab),
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     use_leaf_birnn=args.leaf_birnn,
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
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=test_dataset.collate)
    num_correct = 0
    num_data = len(test_dataset)
    for batch in test_data_loader:
        words = wrap_with_variable(batch['words'], volatile=True,
                                   gpu=args.gpu)
        length = wrap_with_variable(batch['length'], volatile=True,
                                    gpu=args.gpu)
        label = wrap_with_variable(batch['label'], volatile=True,
                                   gpu=args.gpu)
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
    parser.add_argument('--data', required=True)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--leaf-birnn', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
