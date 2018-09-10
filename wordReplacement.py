import pickle

from Experiment import Experiment
from DataGenerator import DataGenerator

from nltk.corpus import gutenberg, words

def generate_data():
    generator = DataGenerator(gutenberg, words.words('en'), 1)
    # for _ in range(10):
    #     print(generator.generate_sentence())
    #     print()

    data = generator.generate(10000)
    with open("data/med_random.pkl", "wb") as outfile:
        pickle.dump(data, outfile)
    print(data)

def tiny_test():
    models = Models()
    exp = Experiment("tiny_random")
    exp.predict(FeaturePipeline.ngram(models.get("nb"), 1))

if __name__ == "__main__":
    # generate_data()
    with open("data/med_random.pkl", "rb") as infile:
        loaded_data = pickle.load(infile)
        print(loaded_data)