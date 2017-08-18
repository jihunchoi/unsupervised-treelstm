import numpy as np


def load_glove(path, vocab, init_weight: np.array):
    word_vectors = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, *values = line.split()
            try:
                if vocab.has_word(word):
                    if word in word_vectors:
                        # Let's use the first occurrence only.
                        continue
                    word_vector = np.array([float(v) for v in values])
                    word_vectors[word] = word_vector
            except ValueError:
                # 840D GloVe file has some encoding errors...
                # I think they can be ignored.
                continue
    glove_weight = np.zeros_like(init_weight)
    for word in word_vectors:
        word_index = vocab.word_to_id(word)
        glove_weight[word_index, :] = word_vectors[word]
    return glove_weight
