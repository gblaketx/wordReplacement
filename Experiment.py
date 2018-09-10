import pickle, FeaturePipeline

from Models import Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score 

class Experiment:
    SPLIT_RAND_STATE = 41

    def __init__(self, dataset, splits=["train", "dev"]):
        # TODO: enforce splits length
        train_data, test_data = self.loadData(dataset, splits)

        self.x_train, self.y_train = self.split_data(train_data)
        self.x_test, self.y_test = self.split_data(test_data)

    def predict(self, model):
        fit_model = model.fit(self.x_train, self.y_train)
        predictions = fit_model.predict(self.x_test)
        print("")
        print("Accuracy: {}".format(accuracy_score(self.y_test, predictions)))
        print(classification_report(self.y_test, predictions, digits=3))

    def loadData(self, name, splits):
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

def test():
    models = Models()

    exp = Experiment("small_random")
    exp.predict(FeaturePipeline.ngram(models.get("nb"), 2))

if __name__ == "__main__":
    test()