import nltk
import numpy as np
from nltk import tree
from tqdm import tqdm


class ProbabilisticLexicon(object):

    def __init__(self, lexical_rules):
        self.words = {}

        # fill self.words
        for rules in lexical_rules:
            assert len(rules.rhs()) == 1
            tag, token = rules.lhs(), rules.rhs()[0]
            if token not in self.words:
                self.words[token] = {'tag': {},
                                     'nb': 0}
            if tag not in self.words[token]['tag']:
                self.words[token]['tag'][tag] = 0
            self.words[token]['tag'][tag] += 1
            self.words[token]['nb'] += 1

        # convert count to probabilities
        for token in self.words:
            n = self.words[token]['nb']
            for tag in self.words[token]['tag']:
                self.words[token]['tag'][tag] /= n

    def get_k_tag(self, word, k, log=True):
        """ Return dictionary of k most probable tag associated to given word """
        return {ke: (np.log(v) if log else v) for ke, v in
                sorted(self.words[word]['tag'].items(), key=lambda item: item[1], reverse=True)[:k]}

    def __len__(self):
        """ return number of triplets (token, tag, proba) """
        return sum(map(lambda x: len(self.words[x]["tag"]), self.words))

    def __repr__(self):
        """ represent as triplets (token, tag, proba) """
        rep = ""
        for word in self.words:
            for tag, prob in self.words[word]["tag"].items():
                rep += "'{}', {}, {:g}\n".format(word, tag, prob)
        return rep


def load_bracketed_data(path):
    with open(path, 'r') as f:
        raw_data = f.read().split('\n')

        def remove_useless_parenthesis(bracketed_expr):
            while '( ' in bracketed_expr and bracketed_expr.find('( ') == 0:
                bracketed_expr = bracketed_expr[bracketed_expr.find('(') + 1:bracketed_expr.rfind(')')]
            return bracketed_expr

        raw_data = list(map(remove_useless_parenthesis, raw_data))  # remove first and last parenthesis
        return raw_data


def get_to_predict_data(filepath, start=0, end=1):
    """ load raw token data from file (path) """
    with open(filepath, 'r') as f:
        raw_data = f.read().split('\n')
    raw_data = list(map(str.split, raw_data))
    start = int(start * len(raw_data))
    end = int(end * len(raw_data))
    assert start < end
    return raw_data[start:end]


def get_tree_data_from_parsed(path, start=0, end=1):
    def get_tree_data(str_data):
        return map(lambda s: tree.Tree.fromstring(s), str_data)

    print("Loading data...")
    bracketed_data = load_bracketed_data(path)
    n = len(bracketed_data)
    start = int(n * start)
    end = int(n * end)
    assert start < end, (start, end)
    train_data = bracketed_data[start:end]

    print("Generating trees...")
    train_tree_data = list(get_tree_data(train_data))

    return train_tree_data


def get_to_eval_data(path, start=0, end=1):
    eval_tree_data = get_tree_data_from_parsed(path, start=start, end=end)
    print("Generate validation and test datasets...")
    sentences, pos_tags = aux_tree_data(eval_tree_data)
    return sentences, pos_tags


def get_train_data(path, start=0, end=1):
    train_tree_data = get_tree_data_from_parsed(path, start=start, end=end)

    print("Get train vocabulary...", end=" ")
    train_vocabulary = get_vocabulary(train_tree_data)
    print(f"{len(train_vocabulary)} words")

    print("Separate grammar from lexical rules...")
    train_grammar_rules = []
    train_lexical_rules = []

    for tree_h in tqdm(train_tree_data):
        nltk.treetransforms.collapse_unary(tree_h, collapsePOS=True, joinChar='@')
        nltk.treetransforms.chomsky_normal_form(tree_h, horzMarkov=2)
        for production in tree_h.productions():
            if production.is_nonlexical():
                assert all(map(lambda x: type(x) == nltk.Nonterminal, production.rhs()))
                train_grammar_rules.append(production)
            else:
                assert len(production) == 1
                train_lexical_rules.append(production)

    print("Get probabilistic lexicon...", end=" ")
    train_prob_lexicon = ProbabilisticLexicon(train_lexical_rules)
    print(f"{len(train_prob_lexicon)} triplets (token, tag, prob)")

    print("Get grammar pcfg...", end=" ")
    train_grammar_rules = nltk.induce_pcfg(nltk.Nonterminal('SENT'), train_grammar_rules)
    print(f"{len(train_grammar_rules.productions())} productions")

    train_rhs_index = build_rhs_index(train_grammar_rules)

    # we have binarized grammar rules
    assert train_grammar_rules.is_binarised()

    # the only unitary rules involve the starting element of grammar
    assert all(map(lambda rule: rule.lhs() == train_grammar_rules.start(),
                   [r for r in train_grammar_rules.productions() if len(r.rhs()) == 1]))

    print("Done")
    return train_vocabulary, train_grammar_rules, train_rhs_index, train_prob_lexicon


def build_rhs_index(grammar_rules):
    """ from grammar rules build dictionary allowing fast access to production with given rhs """

    prod_index_rhs = {}
    for i, prod in enumerate(grammar_rules.productions()):
        rhs = prod.rhs()
        if rhs not in prod_index_rhs:
            prod_index_rhs[rhs] = set()
        prod_index_rhs[rhs].add(prod)
    for rhs in prod_index_rhs:
        prod_index_rhs[rhs] = sorted(prod_index_rhs[rhs], key=lambda r: r.prob(), reverse=True)
    return prod_index_rhs


def preprocess_tree_labels(tree_h):
    """ Remove hyphens from tree labels (Tag)

    Parameters
    ----------
    tree_h: nlkt.Tree
        tree to process

    """

    assert type(tree_h) == tree.Tree, type(tree_h)

    i = str.find(tree_h.label(), '-')
    if i >= 0:
        assert i > 0
        tree_h.set_label(tree_h.label()[:i])

    if type(tree_h[0]) == tree.Tree:
        for child in tree_h:
            preprocess_tree_labels(child)

    return


def split_tag_token(tree_data):
    """ Take a nlkt.Tree and return two lists one containing terminal Tag and the other the corresponding token """

    Tags, tokens = [], []
    if type(tree_data[0]) == str:
        Tags.append(tree_data.label())
        tokens.append(tree_data[0])
        return Tags, tokens

    for child_tree in tree_data:
        child_Tags, child_tokens = split_tag_token(child_tree)
        Tags.extend(child_Tags)
        tokens.extend(child_tokens)

    return Tags, tokens


def aux_tree_data(tree_data):
    """ Turn list of trees corresponding to test data into list of test sentences and corresponding labels
        Remove hyphens from Tag
    """

    labels, sentences = [], []
    for tree_ in tree_data:
        preprocess_tree_labels(tree_)  # remove hyphens from Tag
        Tags, tokens = split_tag_token(tree_)
        labels.append(Tags)
        sentences.append(tokens)

    return sentences, labels


def get_vocabulary(tree_data):
    """ From trees get a set corresponding to the lexical vocabulary """
    sentences, _ = aux_tree_data(tree_data)
    vocabulary = set()
    for sentence in sentences:
        vocabulary.update(sentence)
    return vocabulary


def get_pos_tags(parsing_tree):
    """ Get the PoS tag from given tree (DFS) """
    assert type(parsing_tree) == nltk.tree.Tree, type(parsing_tree)
    PoS_tags = []
    to_deal = [parsing_tree]
    while len(to_deal) > 0:
        current_tree = to_deal.pop()
        if len(current_tree) == 1 and type(current_tree[0]) == str:
            PoS_tags.append(current_tree.label())
        else:
            to_deal.extend(current_tree[::-1])
    return PoS_tags
