import numpy as np


def load_glove(path, vocab, init_weight: np.array):
    glove_weight = init_weight.copy()
    with open(path, 'r', encoding='latin1') as f:
        for line in f:
            word, *values = line.split()
            if vocab.has_word(word):
                values = np.array([float(v) for v in values])
                glove_weight[vocab.word_to_id(word), :] = values
    return glove_weight
