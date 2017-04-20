#coding:utf-8
#adabooooooooooooooooooooooooooooooost

import numpy as np
np.random.seed(114514)
from sklearn.ensemble import AdaBoostClassifier

class adaboost():
    def __init__(self):
        self.model_create()
    def model_create(self):
        self.model = AdaBoostClassifier(n_estimators=50,random_state=114514)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        return self.model.predict_proba(x)