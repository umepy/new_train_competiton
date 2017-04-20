#coding:utf-8
#booooooooooooooooooooooooooooooost

from sklearn.ensemble import GradientBoostingClassifier

class gboost():
    def __init__(self,warm):
        self.model_create(warm)
    def model_create(self,warm):
        self.model = GradientBoostingClassifier(n_estimators=30,random_state=114514,warm_start=warm)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        return self.model.predict_proba(x)