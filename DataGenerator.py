### Data Generator
# Generates word replacement data

import nltk.corpus, random, string

from enum import Enum
from nltk import ne_chunk, pos_tag

BLANK_TAG = "<BLANK/>"

class DataGenerator:
    STOPWORDS = frozenset(nltk.corpus.stopwords.words('english') + ["mr", "mrs", "ms", "miss", "dr", "sir", "oh", "th", "said"])
    PUNCTUATION = frozenset(string.punctuation)
    MIN_SENTENCE_LENGTH = 10
    NUM_DISTRACTORS = 4
    # Default sentence_corpus Project Gutenberg Selections

    def __init__(self, sentence_corpus, replacement_word_corpus, seed):

        self.corpus = sentence_corpus
        self.replacements = [word for word in replacement_word_corpus if word.lower() not in DataGenerator.STOPWORDS]
        self.distractorTypes = Distractors.RANDOM
        random.seed(seed)

    def generate(self, n, unique_sentences=True):
        sentences = []
        used = set()

        if unique_sentences:
            while len(sentences) < n:
                candidate = self.generate_sentence()
                if candidate.tokens in used:
                    continue
                used.add(candidate.tokens)
                sentences.append(self.generate_sentence())
        else:
            for _ in range(n):
                sentences.append(self.generate_sentence())
        return sentences

    def generate_sentence(self):
        blankIndex = [-1]
        while blankIndex[0] < 0:
            # Pick a text from fieldID, then a sentence from the text
            text = random.choice(self.corpus.fileids())
            sentences = self.corpus.sents(text)
            chosen_sent = random.choice(sentences)
            if len(chosen_sent) < DataGenerator.MIN_SENTENCE_LENGTH:
                continue
            blankIndex = DataGenerator.pick_blank_index(chosen_sent)

        return ReplacementSentence(chosen_sent, blankIndex, [self.choose_distractors(chosen_sent, blankIndex)])

    def choose_distractors(self, chosen_sent, blankIndex):
        """

        @param blankIndex a SINGLE blank index
        """
        distractors = []
        if self.distractorTypes == Distractors.RANDOM:
            while len(distractors) < DataGenerator.NUM_DISTRACTORS:
                distractor = random.choice(self.replacements)
                if distractor not in distractors:
                    distractors.append(distractor)
        else:
            raise NotImplementedError("Distractor type: {} is not supported".format(self.distractorTypes))

        return distractors

    @staticmethod
    def pick_blank_index(sent):
        """
        @return a list containing a single blank index
        """
        validIndices = set(range(len(sent)))
        namedEntities = frozenset(map(lambda elem: elem[0][0], ne_chunk(pos_tag(sent), binary=True).subtrees(filter=lambda x: x.label() == "NE")))
        print("Named Entities: {}".format(namedEntities))
        
        for i, token in enumerate(sent):
            if token.lower() in DataGenerator.STOPWORDS or token.isnumeric() or token[0] in DataGenerator.PUNCTUATION or token in namedEntities:
                # print("Removing token: {}, index: {}".format(token, i))
                validIndices.remove(i)
        for v in validIndices:
            print(sent[v], end=' ')
        print()
        # If we've filtered out all the words, return a sentinel value
        if len(validIndices) == 0: 
            return [-1]
        
        return random.sample(validIndices, 1)

class ReplacementSentence:

    def __init__(self, tokens, blankIndices, distractors):

        # A tuple of indices into self.tokens marked with the special <BLANK/> tag
        self.blankIndices = tuple(blankIndices)

        # A tuple of tokens the sentence contains, in order, including the answers
        self.tokens = tuple(tokens)

        # List of lists of distractor words for each repacement index
        self.distractors = distractors

        # A list of indices (integers) at which the answer appears in candidates
        self.answerIndices = tuple(random.randint(0, DataGenerator.NUM_DISTRACTORS) for _ in blankIndices)

    def __repr__(self):
        return "{{Sentence: {}, BlankIndices: {}, Distractors:{}, AnswerIndices:{}}}".format(self.tokens, self.blankIndices, self.distractors, self.answerIndices)

    def getBlankedSentence(self):
        """
        @return a list of tokens with the <BLANK/> tag at the blankIndices
        """
        return [BLANK_TAG if i in self.blankIndices else token for i, token in enumerate(self.tokens)]

    def getAnswerWords(self):
        """
        @return a list of answer tokens, in the order in which they appear in the sentence
        """
        return tuple(self.tokens[i] for i in self.blankIndices)

    def getAnswerIndex(self, i):
        """
        @return the index in candidates at which answer appears
        """
        return self.answerIndices[i]

    def getCandidates(self, i):
        # TODO: should this be shuffled?
        # candidates = self.distractors.copy()
        # candidates.append(self.tokens[self.blankIndices[i]])
        # return candidates
        # return self.distractors[i] + [self.tokens[self.blankIndices[i]]]

        answerIndex = self.getAnswerIndex(i)
        return self.distractors[i][:answerIndex] + [self.tokens[self.blankIndices[i]]] + self.distractors[i][answerIndex:]

    def __eq__(self, other):
        """
        @override
        @return true if ReplacementSentences are strictly equal
        """
        if isinstance(other, ReplacementSentence):
            return self.tokens == other.tokens and self.blankIndices == other.blankIndices
        return False

class SupportedCorpora(Enum):
    GUTENBERG = 1

    @staticmethod
    def from_string(name):
        if (name.lower() == "gutenberg"):
            return SupportedCorpora.GUTENBERG
        else:
            raise ValueError("Unsupported corpus type {}".format(name))

class Distractors(Enum):
    RANDOM = 1
    MATCH_POS = 2
    MATCH_POS_AND_SIMILAR = 3
    FROM_SENTENCE = 4