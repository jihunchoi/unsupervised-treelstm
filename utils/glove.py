import numpy as np


def load_glove(path, vocab, init_weight: np.array):
    glove_weight = init_weight.copy()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, *values = line.split()
            try:
                if vocab.has_word(word):
                    values = np.array([float(v) for v in values])
                    glove_weight[vocab.word_to_id(word), :] = values
            except ValueError:
                # 840D GloVe file has some encoding errors...
                # I think they can be ignored
                continue
    return glove_weight
