import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    """
        Cosine similarity reflects the degree of similariy between u and v

        Arguments:
            u -- a word vector of shape (n,)
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    distance = 0.0
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(np.square(u)))
    norm_v = np.sqrt(np.sum(np.square(v)))
    similarity = dot / (norm_u * norm_v)

    return similarity


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]
apple = word_to_vec_map["apple"]
banana = word_to_vec_map["banana"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))
print("cosine_similarity(apple, banana) = ", cosine_similarity(apple, banana))


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
        Performs the word analogy task as explained above: a is to b as c is to ____.

        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors.

        Returns:
        best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None

    for w in words:
        if w in [word_a, word_b, word_c]:
            continue
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


triads_to_try = [('italy', 'italian', 'china')]
for triad in triads_to_try:
    print('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))
