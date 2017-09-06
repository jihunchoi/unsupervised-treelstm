import argparse

from sst.utils.dataset import read_data


def collect_words(data):
    word_set = set()
    for words, label in data:
        word_set.update(words)
    return word_set


def save_vocab(word_set, path):
    with open(path, 'w', encoding='utf-8') as f:
        for word in word_set:
            f.write(f'{word}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    data_paths = args.data_paths.split(':')
    data_paths = [p for p in data_paths if p.strip()]
    data = []
    for data_path in data_paths:
        data.extend(read_data(path=data_path))
    word_set = collect_words(data)
    save_vocab(word_set=word_set, path=args.out)


if __name__ == '__main__':
    main()
