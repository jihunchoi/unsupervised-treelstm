import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from snli.model import SNLIModel
from snli.utils.dataset import SNLIDataset


def evaluate(args):
    with open(args.data, 'rb') as f:
        test_dataset: SNLIDataset = pickle.load(f)
    word_vocab = test_dataset.word_vocab
    label_vocab = test_dataset.label_vocab
    model = SNLIModel(num_classes=len(label_vocab), num_words=len(word_vocab),
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      intra_attention=args.intra_attention,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional)
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=test_dataset.collate)
    num_correct = 0
    num_data = len(test_dataset)
    for batch in test_data_loader:
        pre = batch['pre'].to(args.device)
        hyp = batch['hyp'].to(args.device)
        pre_length = batch['pre_length'].to(args.device)
        hyp_length = batch['hyp_length'].to(args.device)
        label = batch['label'].to(args.device)
        logits = model(pre=pre, pre_length=pre_length,
                       hyp=hyp, hyp_length=hyp_length)
        label_pred = logits.max(1)[1]
        num_correct_batch = torch.eq(label, label_pred).long().sum()
        num_correct_batch = num_correct_batch.item()
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
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
