import argparse

import jsonlines
from nltk import word_tokenize


def collect_words(path):
    word_set = set()
    with jsonlines.open(path, 'r') as reader:
        for obj in reader:
            for key in ['sentence1', 'sentence2']:
                words = word_tokenize(obj[key].lower())
                word_set.update(words)
    return word_set


def save_vocab(word_set, path):
    with open(path, 'w', encoding='utf-8') as f:
        for word in word_set:
            f.write(f'{word}\t0\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    data_paths = args.data_paths.split(':')
    data_paths = [p for p in data_paths if p.strip()]
    word_set = set()
    for data_path in data_paths:
        word_set = word_set | collect_words(data_path)
    save_vocab(word_set=word_set, path=args.out)


if __name__ == '__main__':
    main()
