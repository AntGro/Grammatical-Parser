import numba
import numpy as np

from utils.utils import levenshtein_thr, get_embedding


def cosine_nearest(word_embedding, embedding_table, id_word_dic):
    """ Compute nearest word according to cosine metric

    Parameters
    ----------
    :param word_embedding: np.array
        embedding of the word of interset
    :param embedding_table: numpy.array
        table of embeddings (one line for one NORMALIZED embedding)
    :param id_word_dic: dict
        dictionary mapping ids (int) to words (str)

    Returns
    -------
    closest_word: str
        closest word to entry word w according to cosine metric
    """
    closest_word = id_word_dic[np.argmax(embedding_table @ word_embedding)]
    return closest_word


def oov(word, train_voc,
        fr_voc, all_embs, all_word_id_dic, voc_embs, voc_id_word_dic,
        transformations,
        k=2):
    candidates = set()
    assert word not in train_voc
    train_candidates = oov_levenshtein_normalize(word, train_voc, k=k, transformations=transformations)
    candidates.update(train_candidates)

    fr_candidates = oov_levenshtein_normalize(word, fr_voc, k=k, transformations=transformations)
    for fr_candidate in fr_candidates:
        candidates.add(
            oov_embedding(fr_candidate, all_embs=all_embs, all_word_id_dic=all_word_id_dic,
                          voc_embs=voc_embs, voc_id_word_dic=voc_id_word_dic))
    return candidates


def oov_levenshtein_normalize(word, vocabulary, k, transformations):
    best_candidates, best_dist = oov_levenshtein(word, list(vocabulary), k=k)
    for transfo in transformations:
        word_t = transfo(word)
        candidates, dist = oov_levenshtein(word_t, list(vocabulary), k=k, best_dist=best_dist - 1)
        if dist <= best_dist - 1:
            if dist < best_dist - 1:
                best_candidates = candidates
                best_dist = dist + 1  # take the subtitution operation into account
            else:
                best_candidates.extend(candidates)
    return list(set(best_candidates))[:k]


@numba.njit
def oov_levenshtein(word, vocabulary, k=2, best_dist=np.inf):
    """ Find @k closest words from @word in @vocabulary """
    best_candidates = ['']  # specify list type for numba
    best_candidates.pop()
    new_list = True
    for candidate in vocabulary:
        dist = levenshtein_thr(word, candidate, best_dist)  # return -1 is distance greater than best distance
        if 0 <= dist <= best_dist:
            if dist < best_dist or new_list:
                best_candidates = [candidate]
            elif len(best_candidates) < k:
                best_candidates.append(candidate)
            new_list = len(best_candidates) == k
            best_dist = dist - new_list
    if len(best_candidates) == 0:
        best_dist = np.inf
    return best_candidates, best_dist + new_list


def oov_embedding(word, all_embs, all_word_id_dic, voc_embs, voc_id_word_dic):
    word_emb = get_embedding(word, all_embs, all_word_id_dic)
    return cosine_nearest(word_emb, voc_embs, voc_id_word_dic)
