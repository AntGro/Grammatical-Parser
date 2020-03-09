import pickle

import numba
import numpy as np
from nltk import tree
from tqdm import tqdm

from utils.cyk import cyk
from utils.data_process import get_pos_tags


@numba.njit
def levenshtein(s1, s2):
    m = np.zeros((len(s1) + 1, len(s2) + 1), dtype=numba.types.int16)
    m[:, 0] = np.arange(len(s1) + 1)
    m[0] = np.arange(len(s2) + 1)
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            m[i, j] = min(min(m[i - 1, j] + 1, m[i, j - 1] + 1), m[i - 1, j - 1] + int(s1[i - 1] != s2[j - 1]))
    return m[-1, -1]


@numba.njit
def levenshtein_thr(s1, s2, thr=np.inf):
    if len(s2) > len(s1):
        s1, s2 = s2, s1
    if len(s1) - len(s2) > thr:
        return -1
    m = np.zeros((len(s1) + 1, len(s2) + 1), dtype=numba.types.int16)
    m[:, 0] = np.arange(len(s1) + 1)
    m[0] = np.arange(len(s2) + 1)
    d2 = 1
    for i in range(2, len(s1) + len(s2) + 1):
        d1 = d2
        d2 = np.inf
        if i <= len(s1):
            d2 = i
        for k in range(max(1, i - len(s2)), min(len(s1) + 1, i)):
            j = i - k
            m[k, j] = min(min(m[k - 1, j] + 1, m[k, j - 1] + 1), m[k - 1, j - 1] + int(s1[k - 1] != s2[j - 1]))
            d2 = min(d2, m[k, j])
        if min(d1, d2) > thr:
            return -1
    return m[-1, -1]


def get_embeddings(path, do_normalize=True):
    """ Load embeddings from file and build associated dictionary
    :rtype: words: set
        set of embedded words
    :rtype: embeddings: np.array
        embedding table (one line per word)
    :rtype: word_id: dict
        dictionary word -> embedding id
    :rtype: id_word: dict
        dictionary embedding id -> word
    """
    words, embeddings = pickle.load(open(path, 'rb'), encoding='latin1')

    # normalize embeddings
    if do_normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    print("Emebddings shape is {}".format(embeddings.shape))

    # Map words to indices and vice versa
    word_id = {w: i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))
    return words, embeddings, word_id, id_word


def process_embeddings(word_embeddings, word_id_dic, vocabulary, re_rules):
    """ Extract embeddings of words in vocabulary and return them along with associated dictionaries """
    selected_word_embeddings = []
    selected_word_id_dic = {}
    selected_id_word_dic = {}
    i = 0
    for word in vocabulary:
        candidate = word if word in word_id_dic else normalize(word, word_id_dic, re_rules)
        if candidate in word_id_dic:
            selected_word_embeddings.append(word_embeddings[word_id_dic[candidate]])
            selected_word_id_dic[word] = i
            selected_id_word_dic[i] = word
            i += 1

    return np.array(selected_word_embeddings), selected_word_id_dic, selected_id_word_dic


def case_normalizer(word, dictionary):
    """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative. """
    w = word
    lower = (dictionary.get(w.lower(), 1e12), w.lower())
    upper = (dictionary.get(w.upper(), 1e12), w.upper())
    title = (dictionary.get(w.title(), 1e12), w.title())
    results = [lower, upper, title]
    results.sort()
    index, w = results[0]
    if index != 1e12:
        return w
    return word


def normalize(word, word_id_dic, re_rules):
    """ Find the closest alternative in case the word is OOV."""
    i = 0
    while word not in word_id_dic and i < len(re_rules):
        word = re_rules[i](word)
        i += 1
    if word not in word_id_dic:
        word = case_normalizer(word, word_id_dic)
    return word


def get_embedding(w, embedding_table, word_id_dic):
    """ Compute embedding of given word based on an embedding table """
    assert w in word_id_dic
    return embedding_table[word_id_dic[w]]


def evel_prediction(labels, true_labels):
    assert len(labels) == len(true_labels), labels
    return np.mean(np.array(labels) == np.array(true_labels))


def evaluate(sentences, true_labels, p_gram_rules, p_lexicon, rhs_index,
             oov_handler, p_output=False, beam=10, chrono=False):
    if type(sentences[0]) != list:
        sentences, true_labels = [sentences], [true_labels]
    assert type(true_labels[0]) == type(sentences[0])

    returns = predict(sentences, p_gram_rules, p_lexicon, rhs_index, oov_handler, p_output, beam, chrono)
    predictions, parsed_rate = returns[:2]
    score = 0
    for j in range(len(sentences)):
        pos_tags = get_pos_tags(tree.Tree.fromstring(predictions[j]))
        score += evel_prediction(pos_tags, true_labels[j])

    score /= len(sentences)
    if p_output:
        print(f"- PoS tags accuracy: {score * 100:.2f}%")

    return predictions, parsed_rate, score


def predict(sentences, p_gram_rules, p_lexicon, rhs_index,
            oov_handler, p_output=False, beam=10, chrono=False):
    if type(sentences[0]) != list:
        sentences = [sentences]

    if chrono:
        t1, t2 = np.zeros(len(sentences)), np.zeros(len(sentences))
    parsed = 0
    parsing_bracketed_str = []
    for j, sentence in tqdm(enumerate(sentences)):
        results = cyk(sentence, p_gram_rules, p_lexicon, rhs_index, oov_handler=oov_handler,
                      beam=beam, chrono=chrono)
        if chrono:
            t1[j] = results[2]
            t2[j] = results[3]
        parsing_tree, total_parse = results[:2]
        parsing_tree.un_chomsky_normal_form(unaryChar='@')
        parsed += total_parse

        parsing_bracketed_str.append('( ' + parsing_tree._pformat_flat('', '()', False) + ')\n')
        if p_output:
            if not total_parse:
                print("/!\\ PARTIAL PARSING /!\\", end="\t")
            print(parsing_tree._pformat_flat('', '()', False))  # print bracketted expression
    parsed /= len(sentences)
    to_return = [parsing_bracketed_str, parsed]
    if p_output:
        print(
            f"Parsed {len(sentences)} sentences\n. - Total parsing: {parsed * 100:.2f}% ")
        if chrono:
            print(
                f"  - Mean time to handle oov: {np.mean(t1):.2f}s\n  - Mean time to build membership table: {np.mean(t2):.2f}s")
    return to_return


def write_in_file(path, sentences):
    with open(path, 'w') as f:
        for i, sentence in enumerate(sentences):
            if i < len(sentences) - 1:
                f.write(sentence)
            else:
                if '\n' in sentence:
                    f.write(sentence[:sentence.rfind('\n')])
                else:
                    f.write(sentence)
    f.close()
