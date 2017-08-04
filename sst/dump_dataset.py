import argparse
import pickle

from sst.utils.dataset import SSTDataset
from utils.vocab import Vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--vocab-size', required=True, type=int)
    parser.add_argument('--max-length', default=None, type=int)
    parser.add_argument('--binary', default=False, action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    word_vocab = Vocab.from_file(path=args.vocab, add_pad=True, add_unk=True,
                                 max_size=args.vocab_size)
    data_reader = SSTDataset(
        data_path=args.data, word_vocab=word_vocab, max_length=args.max_length,
        binary=args.binary)
    with open(args.out, 'wb') as f:
        pickle.dump(data_reader, f)


if __name__ == '__main__':
    main()
