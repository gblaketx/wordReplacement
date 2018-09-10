
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge

class Models:
    
    def __init__(self):
        self.models = {
            'nb' : MultinomialNB, 
            'linearReg' : LinearRegression,
            'sgd' : SGDRegressor,
            'ridge' : Ridge,
            'svr' : SVR
        }

    def get(self, name):
        print("\nModel: {}".format(name))
        return self.models[name]()