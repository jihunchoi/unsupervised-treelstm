import re

import numpy as np
import torch
from torch.utils.data import Dataset


def read_data(path, max_length=None):
    word_pattern = re.compile(r'([^()\s]+)\)')
    data = []
    with open(path, 'r') as f:
        for line in f:
            label = int(line[1])
            words = [w.lower() for w in word_pattern.findall(line)]
            if max_length is not None and len(words) > max_length:
                continue
            data.append((words, label))
    return data


class SSTDataset(Dataset):

    def __init__(self, data_path, word_vocab, max_length, binary):
        self.word_vocab = word_vocab
        self.binary = binary
        self.num_classes = 2 if binary else 5
        self._max_length = max_length
        data = read_data(data_path, max_length)
        self._data = self._preprocess_data(data)

    def _preprocess_data(self, data):
        preprocessed = []
        for words, label in data:
            if self.binary:
                if label < 2:
                    label = 0
                elif label > 2:
                    label = 1
                else:
                    continue
            words = [self.word_vocab.word_to_id(w) for w in words]
            preprocessed.append((words, label))
        return preprocessed

    def _omit_words(self, words, omit_prob):
        words = words.copy()
        length = len(words)
        num_omits = round(length * omit_prob)
        random_inds = np.random.permutation(length)[:num_omits]
        for i in random_inds:
            words[i] = self.word_vocab.unk_id
        return words

    def _pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch, omit_prob=0.0):
        words_batch, label_batch = list(zip(*batch))
        length_batch = [len(d) for d in words_batch]
        if omit_prob > 0:
            words_batch = [self._omit_words(words=words, omit_prob=omit_prob)
                           for words in words_batch]
        words_batch = self._pad_sentence(words_batch)
        words = torch.LongTensor(words_batch)
        length = torch.LongTensor(length_batch)
        label = torch.LongTensor(label_batch)
        return {'words': words, 'length': length, 'label': label}
