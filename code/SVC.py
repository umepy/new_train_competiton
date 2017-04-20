#coding:utf-8
#SVM de Learn

from sklearn.svm import SVC

class gboost():
    def __init__(self):
        self.model_create()
    def model_create(self):
        self.model = SVC(random_state=114514,probability=True,class_weight='balanced')
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        return self.model.predict_proba(x)