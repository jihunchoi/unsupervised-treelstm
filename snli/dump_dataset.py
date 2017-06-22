import argparse
import pickle

from snli.util.dataset import SNLIDataset
from utils.vocab import Vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--vocab-size', required=True, type=int)
    parser.add_argument('--max-length', required=True, type=int)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    word_vocab = Vocab.from_file(path=args.vocab, add_pad=True, add_unk=True,
                                 max_size=args.vocab_size)
    label_dict = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    label_vocab = Vocab(vocab_dict=label_dict, add_pad=False, add_unk=False)
    data_reader = SNLIDataset(
        data_path=args.data, word_vocab=word_vocab, label_vocab=label_vocab,
        max_length=args.max_length)
    with open(args.out, 'wb') as f:
        pickle.dump(data_reader, f)


if __name__ == '__main__':
    main()
