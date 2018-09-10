from Featurizers import *

from sklearn.base import BaseEstimator, TransformerMixin

class NGramExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, n):
        self.n = n
        print("[{}-grams]".format(n), end=' ')

    def transform(self, examples, y=None):
        return [ngram_feats(ex, self.n) for ex in examples]

    def fit(self, examples, y=None):
        return self


class ChoiceIndexExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, i):
        self.i = i
        print("[Choice Index]", end=' ')

    def transform(self, examples, y=None):
        return [{choice: index for index, choice in enumerate(ex.getCandidates(self.i))} for ex in examples]

    def fit(self, examples, y=None):
        return self