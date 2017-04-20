#coding:utf-8
#KNNを用いて予測を行うクラス

from sklearn.neighbors import KNeighborsClassifier

class KNN():
    def __init__(self):
        self.model_create()
    def model_create(self):
        self.model = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        return self.model.predict_proba(x)