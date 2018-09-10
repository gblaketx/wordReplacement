import Extractors

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# class FeaturePipeline:

#     def __init__(self, model):
#         # self.x_train = x_train
#         # self.y_train = y_train
#         self.model = model
#         # self.fit = fit

def createPipeline(model, sent_features, choice_features=None):
    """
    @param sent_features a list of Feature extractor pipelines from the sentence
    """
    if choice_features is None:
        choice_features = getDefaultChoiceFeatures()
        # choice_features = getSingleChoiceFeatures()
    return Pipeline([
        ('feats', FeatureUnion(
            [('choice_feats', choice_features)] + 
            sent_features)),
        ('tfidf', TfidfTransformer()),     #TODO: tf-dif here or on sent_features   
        ('clf', model)
    ])  

def getDefaultChoiceFeatures(i=0):
    return Pipeline([
        ('choice_index', Extractors.ChoiceIndexExtractor(i)),
        ('vect', DictVectorizer())])

def getSingleChoiceFeatures(i=0):
    return Pipeline([
        ('choice_index', Extractors.SingleChoiceExtractor(i)),
        ('vect', DictVectorizer())])    

def ngram(model, ngrams):
    print("Features: ", end='')
    ngram_pipeline = Pipeline([('ngrams', Extractors.NGramExtractor(ngrams)),
                     ('vect', DictVectorizer())])
    return createPipeline(model, [('sent_feats', ngram_pipeline)])

def ngram_reg(model, ngrams):
    print("Features: ", end='')
    ngram_pipeline = Pipeline([('ngrams', Extractors.NGramExtractor(ngrams)),
                     ('vect', DictVectorizer())])
    return createPipeline(model, [('sent_feats', ngram_pipeline)], getSingleChoiceFeatures())