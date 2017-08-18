class Vocab(object):

    def __init__(self, vocab_dict, add_pad, add_unk):
        self._vocab_dict = vocab_dict.copy()
        self._reverse_vocab_dict = dict()
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = len(self._vocab_dict)
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = len(self._vocab_dict)
            self._vocab_dict[self.unk_word] = self.unk_id
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    @classmethod
    def from_file(cls, path, add_pad, add_unk, max_size=None):
        vocab_dict = dict()
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_size and i >= max_size:
                    break
                word = line.strip().split()[0]
                vocab_dict[word] = len(vocab_dict)
        return cls(vocab_dict=vocab_dict, add_pad=add_pad, add_unk=add_unk)

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return len(self._vocab_dict)
