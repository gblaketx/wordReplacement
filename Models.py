
from sklearn.naive_bayes import MultinomialNB

class Models:
    
    def __init__(self):
        self.models = {'nb' : MultinomialNB}

    def get(self, name):
        print("\nModel: {}".format(name))
        return self.models[name]()