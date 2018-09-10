import pickle

from spacy.lang.en import English
from collections import Counter
from nltk.util import ngrams

nlp = English()

START_TOKEN = "<s>"
END_TOKEN = "</s>"

def ngram_feats(example, n, lemmatize=False):
    sentence = [t.lower() for t in example.getBlankedSentence()]

    if lemmatize:
        sentence = [nlp(t)[0].lemma_ for t in sentence]
        
    feats = Counter(sentence)

    if n >= 2:
        feats.update(str(gram) for gram in ngrams(sentence, 2, True, True, START_TOKEN, END_TOKEN))

    if n >= 3:
        feats.update(str(gram) for gram in ngrams(sentence, 3, True, True, START_TOKEN, END_TOKEN))

    return feats

def test():
    with open("data/toy_random.pkl", "rb") as infile:
        data = pickle.load(infile)
        sent = data[0]
        print("############## UNIGRAMS ##############")
        print(ngram_feats(sent, 1))
        print("")

        print("############## BIGRAMS ##############")
        print(ngram_feats(sent, 2))
        print("")

        print("############## TRIGRAMS ##############")
        print(ngram_feats(sent, 3))
        print("")


if __name__ == "__main__":
    test()