#coding:utf-8
#時系列入力化するスクリプト

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

class create_time_data():
    def __init__(self,distance,name,NN):
        self.name = name
        self.distance = distance
        if NN:
            self.read_pickleNN()
        else:
            self.read_pickle()
        self.set_timespan(12)


    def read_pickle(self):
        with open('../data/pickle/'+self.name+'_'+str(self.distance)+'.pickle','rb') as f:
            self.train=pickle.load(f)
    def read_pickleNN(self):
        with open('../data/pickle/'+self.name+'_'+str(self.distance)+'_NN.pickle','rb') as f:
            self.train=pickle.load(f)
    def set_timespan(self,span):
        self.span=span
    def create_data_train(self):
        train_x=[]
        train_y=self.train[['none','people','machine','weather']].ix[self.span:]
        test_x=[]
        for i in tqdm(range(self.span,len(self.train),1)):
            one_step=[]
            for j in range(self.span,0,-1):
                one_step.append(self.train.ix[i-j].tolist()[4:])
            train_x.append(one_step)
        print('\n訓練データ作成完了:\t'+self.name)
        print(np.array(train_x).shape)
        with open('../data/pickle/' + str(self.span) + self.name +'_'+str(self.distance)+'km_x.pickle', 'wb') as f:
            pickle.dump(np.array(train_x), f)
        with open('../data/pickle/' + str(self.span) + self.name +'_'+str(self.distance)+'km_y.pickle', 'wb') as f:
            pickle.dump(np.array(train_y), f)
    def create_data_test(self):
        train_x=[]
        for i in tqdm(range(self.span,len(self.train),1)):
            one_step=[]
            for j in range(self.span,0,-1):
                one_step.append(self.train.ix[i-j].tolist())
            train_x.append(one_step)
        print('\n訓練データ作成完了:\t'+self.name)
        print(np.array(train_x).shape)
        with open('../data/pickle/' + str(self.span) + self.name +'_'+str(self.distance)+'km_x.pickle', 'wb') as f:
            pickle.dump(np.array(train_x), f)



if __name__=='__main__':
    # for i in ['keihintohoku','keiyou','saikyoukawagoe','tyuou','uchibou']:
    #     my=create_time_data(20,i,True)
    #     my.set_timespan(18)
    #     my.create_data_train()
    for i in ['sotobou','syonan','takasaki','utsunomiya','yamanote']:
        my=create_time_data(20,i,False)
        my.set_timespan(18)
        my.create_data_test()