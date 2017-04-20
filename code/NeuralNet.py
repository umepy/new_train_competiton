#coding:utf-8
#ランダムフォレストのクラス

import numpy as np
#seed固定
np.random.seed(114514)
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.callbacks import EarlyStopping,History

class NeuralNetC():
    def __init__(self):
        self.model_create()
        self.ES = EarlyStopping(monitor='val_loss',patience=3)
        self.HS = History()
    def model_create(self):
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=8))
        self.model.add(Activation("relu"))
        self.model.add(Dense(12, input_dim=16))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(8, input_dim=12))
        self.model.add(Activation("relu"))
        self.model.add(Dense(4, input_dim=4))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def fit(self,x,y,size,epoch):
        self.model.fit(x,y,batch_size=size,epochs=epoch,validation_split=0.2,verbose=0,callbacks=[self.ES,self.HS])
    def predict(self,x):
        return self.model.predict_proba(x,batch_size=1,verbose=0)
