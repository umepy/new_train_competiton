#coding:utf-8
#lstmで予測

from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.callbacks import EarlyStopping,History
from keras.layers.recurrent import LSTM

class lstmC():
    def __init__(self):
        self.model_create()
        self.ES = EarlyStopping(monitor='val_loss',patience=3)
        self.HS = History()
    def model_create(self):
        self.model = Sequential()
        self.model.add(LSTM(18, init='uniform', inner_init='uniform', activation='tanh',
                            inner_activation='sigmoid', input_shape=(18, 8),
                            return_sequences=True))
        self.model.add(LSTM(8, init='uniform', inner_init='uniform', activation='tanh',
                            inner_activation='sigmoid'))
        self.model.add(Dense(4, input_dim=8))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def fit(self,x,y,size,epoch):
        self.model.fit(x,y,batch_size=size,epochs=epoch,validation_split=0.2,verbose=0,callbacks=[self.ES,self.HS])
    def predict(self,x):
        return self.model.predict_proba(x,batch_size=1,verbose=0)