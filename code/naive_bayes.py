#coding:utf-8
#ないーぶでどうかな

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

class NaiveBayes():
    def __init__(self):
        self.model_create()
    def model_create(self):
        self.model = BernoulliNB()
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        return self.model.predict_proba(x)