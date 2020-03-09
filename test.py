from utils.utils import levenshtein, levenshtein_thr


def test_levenshtein():
    s1, s2 = 'access', 'across'
    assert levenshtein(s1, s2) == 2, levenshtein(s1, s2)
    s1, s2 = 'logistic', 'regression'
    assert levenshtein(s1, s2) == 7, levenshtein(s1, s2)
    s1, s2 = 'natural', 'language'
    assert levenshtein(s1, s2) == 6, levenshtein(s1, s2)
    print("Test levenshtein_thr function passed!")


def test_levenshtein_thr():
    s1, s2 = 'access', 'across'
    assert levenshtein_thr(s1, s2) == 2, levenshtein_thr(s1, s2)
    s1, s2 = 'logistic', 'regression'
    assert levenshtein_thr(s1, s2) == 7, levenshtein_thr(s1, s2)
    s1, s2 = 'natural', 'language'
    assert levenshtein_thr(s1, s2) == 6, levenshtein_thr(s1, s2)
    s1, s2, thr = 'natural', 'language', 3
    assert levenshtein_thr(s1, s2, thr=thr) == -1, levenshtein_thr(s1, s2, thr=thr)
    s1, s2 = 'risk', 'disease'
    assert levenshtein_thr(s1, s2) == 5, levenshtein_thr(s1, s2)
    s1, s2, thr = 'risk', 'disease', 2
    assert levenshtein_thr(s1, s2, thr=thr) == -1, levenshtein_thr(s1, s2, thr=thr)
    print("Test levenshtein_thr function passed!")


if __name__ == '__main__':
    test_levenshtein()
    test_levenshtein_thr()
