import pickle, FeaturePipeline

from Models import Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score 
from numpy import argmax

class Experiment:
    SPLIT_RAND_STATE = 41

    def __init__(self, dataset, splits=["train", "dev"]):
        # TODO: enforce splits length
        train_data, test_data = Experiment.loadData(dataset, splits)

        self.x_train, self.y_train = self.split_data(train_data)
        self.x_test, self.y_test = self.split_data(test_data)

    def predict(self, model):
        fit_model = model.fit(self.x_train, self.y_train)
        predictions = fit_model.predict(self.x_test)
        print("")
        print("Accuracy: {}".format(accuracy_score(self.y_test, predictions)))
        print(classification_report(self.y_test, predictions, digits=3))

    @staticmethod
    def loadData(name, splits):
        with open("data/{}.pkl".format(name), "rb") as infile:
            data = pickle.load(infile)

            train, rest = train_test_split(data, test_size=0.4, random_state=Experiment.SPLIT_RAND_STATE)
            dev, test = train_test_split(rest, test_size=0.5, random_state=Experiment.SPLIT_RAND_STATE)

            data_splits = {
                'train': train,
                'dev' : dev,
                'test': test
            }

            res = []
            for name in splits:
                res.append(data_splits[name])
            return res

    def split_data(self, data):
        return tuple(zip(*[(s, s.getAnswerIndex(0)) for s in data]))

class RegressionExperiment:

    def __init__(self, dataset, splits=["train", "dev"]):
        # TODO: enforce splits length
        train_data, test_data = Experiment.loadData(dataset, splits)

        self.x_train, self.y_train = self.split_data_regression(train_data)
        self.x_test, self.y_test = self.split_data_reg_bundle_sentences(test_data)


    def predict(self, model):
        fit_model = model.fit(self.x_train, self.y_train)

        predictions = []
        for s in self.x_test:

            s_predictions = fit_model.predict([(s, i) for i, _ in enumerate(s.getCandidates(0))])
            # s_predictions = {i : fit_model.predict([(s, i)]) for i, _ in enumerate(s.getCandidates(0))}
            # print(s_predictions)
            predictions.append(argmax(s_predictions))

        print("")
        print("Accuracy: {}".format(accuracy_score(self.y_test, predictions)))
        print(classification_report(self.y_test, predictions, digits=3))        

    def split_data_regression(self, data):

        # x: (sentence, answer index)
        # y: (1 if answer, 0 otherwise)
        x = []
        y = []

        for s in data:
            for i in range(len(s.getCandidates(0))):
                answerIndex = s.getAnswerIndex(0)
                x.append((s, i))
                y.append(1 if answerIndex == i else 0)

        # for ex in x:
        #     print(ex)
            # if not isinstance(ex, tuple):
            #     print(ex)
        return x, y
        # return tuple(zip(*[((s, i), 1 if s.getAnswerIndex(0) == i else 0) for i in range(len(s.getCandidates(0))) for s in data]))

    def split_data_reg_bundle_sentences(self, data):
        return tuple(zip(*[(s, s.getAnswerIndex(0)) for s in data]))

def test():
    models = Models()

    exp = RegressionExperiment("med_random")
    exp.predict(FeaturePipeline.ngram_reg(models.get("svr"), 1))

if __name__ == "__main__":
    test()