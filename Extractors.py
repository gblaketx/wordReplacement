from Featurizers import *

from sklearn.base import BaseEstimator, TransformerMixin

class NGramExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, n):
        self.n = n
        print("[{}-grams]".format(n), end=' ')

    def transform(self, examples, y=None):
        return [ngram_feats(sent, self.n) for sent, _ in examples]

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

class SingleChoiceExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, i):
        self.i = i
        print("[Single Choice]", end=' ')

    def transform(self, examples, y=None):
        # print("Ex: {}".format(examples[0]))
        # print(examples)

            # print(ex)

        return [{s.getCandidates(self.i)[choice_index]: 1000} for s, choice_index in examples]

    def fit(self, examples, y=None):
        return self
