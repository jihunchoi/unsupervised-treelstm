import argparse
from collections import Counter

import jsonlines
from nltk import word_tokenize


def count_word(path):
    counter = Counter()
    with jsonlines.open(path, 'r') as reader:
        for obj in reader:
            for key in ['sentence1', 'sentence2']:
                words = word_tokenize(obj[key].lower())
                counter.update(words)
    return counter


def save_vocab(counter, path):
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in counter.most_common():
            f.write(f'{k}\t{v}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    word_counter = count_word(args.data)
    save_vocab(counter=word_counter, path=args.out)


if __name__ == '__main__':
    main()
