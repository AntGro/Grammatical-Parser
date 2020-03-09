import time
from collections import Callable
from typing import Dict, Tuple, List

import nltk
import numpy as np

from nltk import tree

from utils.data_process import ProbabilisticLexicon


def cyk(sentence: str,
        p_gram_rules: nltk.grammar.PCFG,
        p_lexicon: ProbabilisticLexicon,
        rhs_index: Dict[Tuple[nltk.Nonterminal, nltk.Nonterminal], List[nltk.grammar.Production]],
        oov_handler: Callable,
        beam: int = 2,
        chrono: object = False):
    """ Apply cyk algorithm
    :param : sentence: str or List[str]
        input sentence
    :param : p_gram_rules: nltk.grammar.PCFG
    :param : p_lexicon: ProbabilisticLexicon
    :param : beam: int
    :param : oov_handler: fct(str) -> str (word in training vocabulary)
    :param : rhs_index: dict(k: (tag, tag), v: list of grammar_rules)
    :param : chrono: boolean
        whether to return execution times for the two steps of cyk
    """

    membership_table = [
        []]  # at index [i][j] will be placed the (at most) best #beam candidates for part of sentence j to j + i + 1
    # stored as a list of triplets (tag, log prob, subtree)

    t = time.time()
    if type(sentence) == str:
        sentence = str.split(sentence)

    # find tag associated to each word
    for i, word in enumerate(sentence):
        membership_table[0].append({})
        if word not in p_lexicon.words:
            candidates = oov_handler(word)
        else:
            candidates = {word}
        for candidate in candidates:
            for k, v in p_lexicon.get_k_tag(candidate, beam, log=True).items():
                if k not in membership_table[0][-1]:
                    membership_table[0][-1][k] = 0
                membership_table[0][-1][k] += v / len(candidates)
        membership_table[0][-1] = [(k, v, tree.Tree(str(k), [word])) for k, v in
                                   sorted(membership_table[0][-1].items(), key=lambda item: item[1], reverse=True)[
                                   :beam]]

    t1 = time.time() - t
    t = time.time()

    # fill the membership table
    for i in range(1, len(sentence)):
        membership_table.append([])
        # fill ith line
        for j in range(len(sentence) - i):
            # fill elements for segment [j, j + i] of size i + 1 -> element [i (length - 1), j (start)]
            log_prob_thr = -np.inf
            best_candidates = []
            for k in range(j, j + i):
                # elements [j, k] | [k + 1, j + i]
                candidates = get_membership(membership_table[k - j][j], membership_table[j + i - k - 1][k + 1],
                                            rhs_index, beam=beam, thr=log_prob_thr,
                                            require_start=(p_gram_rules.start() if i == len(sentence) - 1 else None))
                if len(candidates) > 0:
                    best_candidates.extend(candidates)
                    if len(best_candidates) > 2 * beam:
                        best_candidates = sorted(best_candidates, key=lambda x: x[1], reverse=True)[:beam]
                        log_prob_thr = best_candidates[-1][1]
            membership_table[i].append(best_candidates[:beam])

    t2 = time.time() - t
    no_candidates = len(membership_table[-1][0]) == 0
    if len(sentence) == 1 or no_candidates:
        to_return = [
            nltk.tree.Tree(str(p_gram_rules.start()), [membership_table[0][j][0][-1] for j in range(len(sentence))]),
            len(sentence) == 1]
        if chrono:
            to_return.extend([t1, t2])

    else:
        to_return = [membership_table[-1][0][0][-1],
                     True]  # [-1][0] <- segment [0, sentence_length] ; [0] <- best candidate ; [-1] <- tree
        if chrono:
            to_return.extend([t1, t2])
    return to_return


def get_membership(l_tags, r_tags, rhs_index, beam=2, thr=-np.inf, require_start=None):
    """ Return the #beam most probable triplets (tag, logprob, subtree) for given left_tag and right_tag (rule: tag
    -> left_tag right_tag) """
    best_candidates = []
    for l_tag, l_logprob, l_subtree in l_tags:
        for r_tag, r_logprob, r_subtree in r_tags:
            if (l_tag, r_tag) not in rhs_index:
                continue
            candidates = rhs_index[(l_tag, r_tag)][:beam]  # already sorted in decreasing order
            for candidate in candidates:
                if require_start is not None and require_start != candidate.lhs():
                    continue
                logprob = candidate.logprob() + l_logprob + r_logprob
                tag = candidate.lhs()
                if logprob >= thr:
                    best_candidates.append([tag, logprob, tree.Tree(str(tag), [l_subtree, r_subtree])])
    return best_candidates
