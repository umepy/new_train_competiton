#coding:utf-8
#ランダムフォレストのクラス

import numpy as np
#seed固定
np.random.seed(114514)
from sklearn.ensemble import RandomForestClassifier

class RandomForestC():
    def __init__(self,trees):
        self.model_create(trees)
    def model_create(self,trees):
        self.model = RandomForestClassifier(n_estimators=trees,n_jobs=-1)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        return self.model.predict_proba(x)
