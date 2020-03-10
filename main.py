import re
import warnings

from numba import NumbaPendingDeprecationWarning

from utils.data_process import get_to_eval_data, get_train_data, get_to_predict_data
import argparse

from utils.oov import oov
from utils.utils import get_embeddings, process_embeddings, evaluate, predict, write_in_file


def main():
    text = 'Description of the program arguments'

    parser = argparse.ArgumentParser(description=text)
    parser.add_argument("--train_path", "-t", help="set path where to find training data")
    parser.add_argument("--train_start", help="start data fraction for train data (default 0)")
    parser.add_argument("--train_end", help="end data fraction for train data (default 1)")
    parser.add_argument("--test_path", "-T", help="set path where to find test data")
    parser.add_argument("--test_start", help="start data fraction for test data (default 0)")
    parser.add_argument("--test_end", help="end data fraction for test data (default 1)")
    parser.add_argument("--embedding_path", "-e", help="set path where to find french word embeddings")
    parser.add_argument("--mode", "-m",
                        help="set mode:\n - 'prediction' / 'e': predict only \n - 'evaluation' / 'e': predict and "
                             "evaluate predictions")
    parser.add_argument("--output_path", "-o",
                        help="set path where to write predictions (if None nothing will be written)")
    parser.add_argument("--beam", "-b", help="set beam search size for cyk algorithm (default 10)")

    args = parser.parse_args()

    def change_none(x, val):
        return val if x is None else x

    train_path = args.train_path
    train_start = float(change_none(args.train_start, 0))
    train_end = float(change_none(args.train_end, 1))
    test_path = args.test_path
    test_start = float(change_none(args.test_start, 0))
    test_end = float(change_none(args.test_end, 1))
    embedding_path = args.embedding_path
    mode = args.mode
    beam = int(change_none(args.beam, 10))
    output_path = args.output_path

    assert mode in ('prediction', 'evaluation', 'e', 'p'), mode

    print("#" * 100 + '\n##')
    print('##\t- Build grammar from file: %s' % train_path)
    print('##\t- Build oov module from embeddings stored in: %s' % embedding_path)
    print('##\t- Make {} based on cyk (beam: {}) on sentences in file: {}'.format(
        'predictions' if mode in ('prediction', 'p') else 'evaluations', beam, test_path))
    if output_path is not None:
        print('##\t- Store predictions in file: %s' % output_path)
    else:
        print("## Don't save predictions")
    print('##\n' + "#" * 100)

    train_vocabulary, train_grammar_rules, train_rhs_index, train_unary_dic, train_prob_lexicon = get_train_data(
        train_path, train_start,
        train_end)

    if mode in ('prediction', 'p'):
        test_sentences = get_to_predict_data(test_path, test_start, test_end)
    elif mode in ('evaluation', 'e'):
        test_sentences, test_labels = get_to_eval_data(test_path, test_start, test_end)
    else:
        raise ValueError("Should be 'prediction', 'p', 'evaluation' or 'e', not %s" % mode)
    # load French word embeddings
    fr_words, embeddings, word_id, id_word = get_embeddings(embedding_path)

    # Normalize digits by replacing them with #
    DIGITS = re.compile("[0-9]", re.UNICODE)

    # considered transformations when looking for in vocabulary words
    TRANSFOS = [lambda w: DIGITS.sub("#", w), lambda w: w.lower(), lambda w: w.upper(), lambda w: w.title()]

    train_embeddings, train_word_id, voc_id_word = process_embeddings(word_embeddings=embeddings, word_id_dic=word_id,
                                                                      vocabulary=train_vocabulary,
                                                                      re_rules=[lambda s: DIGITS.sub("#", s)])

    def oov_handler(word):
        return oov(word, train_vocabulary,
                   fr_words, all_embs=embeddings, all_word_id_dic=word_id, voc_embs=train_embeddings,
                   voc_id_word_dic=voc_id_word,
                   transformations=TRANSFOS,
                   k=2)

    print("Vocabulary-specific embedding shape is {}".format(train_embeddings.shape))

    if mode in ('evaluation', 'e'):
        parsed_str, score, parsed = evaluate(test_sentences, test_labels, train_grammar_rules, train_prob_lexicon,
                                             train_rhs_index, train_unary_dic, oov_handler, p_output=True, beam=beam,
                                             chrono=True)
    elif mode in ('prediction', 'p'):
        parsed_str, parsed = predict(test_sentences, train_grammar_rules, train_prob_lexicon, train_rhs_index,
                                     train_unary_dic, oov_handler, p_output=True, beam=beam, chrono=True)
    if output_path is not None:
        print("Write predictions in %s..." % output_path, end=' ')
        write_in_file(output_path, parsed_str)
        print('Done')


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
    main()
