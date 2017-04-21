#coding:utf-8
#与えられた訓練データから検証をする

import numpy as np
np.random.seed(114514)
import pandas as pd
import pickle
import sys
sys.path.append('./code')
from knn import KNN
from adaboost import adaboost
from gboost import gboost
from SVC import SVC
from naive_bayes import NaiveBayes
from xgboost import xgboost
import tqdm


class Validation():
    def __init__(self,dis):
        self.set_distance(dis)
        self.read_data()
        self.loss_sum=0
        self.loss_mean=0
        #number
        self.mymean=[0,0,0,0]
    def set_distance(self,dis):
        self.distance = dis
    def read_data(self):
        self.trains={}
        self.NNtrains = {}
        for i in ['keihintohoku','keiyou','saikyoukawagoe','tyuou','uchibou']:
            with open('../data/pickle/'+i+'_'+str(self.distance)+'.pickle','rb') as f:
                self.trains[i] = pickle.load(f).as_matrix()
        for i in ['keihintohoku','keiyou','saikyoukawagoe','tyuou','uchibou']:
            with open('../data/pickle/'+i+'_'+str(self.distance)+'_NN.pickle','rb') as f:
                self.NNtrains[i] = pickle.load(f).as_matrix()
    def getdata(self,name,num,xy):
        if xy=='x':
            if type(name) == type([]):
                data = self.trains[name[0]]
                for i in range(1,len(name)):
                    data=np.concatenate((data,self.trains[name[i]]),axis=0)
                return data[num:num+5,1:]
            else:
                data = self.trains[name]
                return data[num:num+1,1:]
        elif xy=='y':
            if type(name) == type([]):
                data = self.trains[name[0]]
                for i in range(1, len(name)):
                    data = np.concatenate((data, self.trains[name[i]]), axis=0)
                return data[num+1:num+6,:1]
            else:
                data = self.trains[name]
                return data[num:num + 1, :1]
    def getdataNN(self,name,num,xy):
        if xy=='x':
            if type(name) == type([]):
                data = self.NNtrains[name[0]]
                for i in range(1,len(name)):
                    data=np.concatenate((data,self.NNtrains[name[i]]),axis=0)
                return data[num:num+5,4:]
            else:
                data = self.NNtrains[name]
                return data[num:num+1,4:]
        elif xy=='y':
            if type(name) == type([]):
                data = self.NNtrains[name[0]]
                for i in range(1, len(name)):
                    data = np.concatenate((data, self.NNtrains[name[i]]), axis=0)
                return data[num+1:num+6,:4]
            else:
                data = self.NNtrains[name]
                return data[num:num + 1, :4]
    def getalldata(self,name,num):
        data = self.trains[name[0]]
        for i in range(1, len(name)):
            data = np.concatenate((data, self.trains[name[i]]), axis=0)

        #学習用のダミー訓練データを挿入
        x_data = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
        y_data = np.array([0,1,2,3])
        x_data = np.concatenate((x_data,data[0:num*4,1:]))
        y_data = np.concatenate((y_data,data[4:(num+1)*4,0]))
        return x_data,y_data
    def run(self):
        #５通りの訓練データを作成する
        train_names=[]
        train_names.append(['keiyou','saikyoukawagoe','tyuou','uchibou'])
        train_names.append(['keihintohoku','saikyoukawagoe','tyuou','uchibou'])
        train_names.append(['keihintohoku','keiyou','tyuou','uchibou'])
        train_names.append(['keihintohoku','keiyou','saikyoukawagoe','uchibou'])
        train_names.append(['keihintohoku','keiyou','saikyoukawagoe','tyuou'])
        my = xgboost()
        j = train_names[0]
        result=[]
        #初期値は定数値で与える
        result.append([1,0,0,0])
        self.logloss([1,0,0,0],self.getdata('keihintohoku',0,'y'))
        self.logloss_mean([1, 0, 0, 0], self.getdata('keihintohoku', 0, 'y'))
        one=0
        for i in range(1,len(self.trains['tyuou'])):
            result=[]
            for nn in ['keihintohoku','keiyou','saikyoukawagoe','tyuou','uchibou']:
                self.mymean[np.argmax(self.getdataNN(nn, i - 1, 'y'))] += 1

            if i%5==0:
                x, y = self.getalldata(j, i)
                #x=x.copy(order='C')
                #y=y.copy(order='C')
                my.fit(x, y)
                one=1
            else:
                if one==1:
                    pass
                    #my.fit(self.getdata(j, i, xy='x'), self.getdata(j, i, xy='y'))

            if i%1000==0:
                print(str(i) + ':\t' + str(self.loss_sum / i) + '\t' + str(self.loss_mean / i))

            if one!=0:
                self.logloss(my.predict(self.getdata('keihintohoku',i,'x')), self.getdataNN('keihintohoku',i,'y'))
                self.logloss_mean(self.mymean/np.mean(self.mymean)/4.0, self.getdataNN('keihintohoku',i,'y'))
            else:
                self.logloss(self.mymean / np.mean(self.mymean) / 4.0,self.getdataNN('keihintohoku', i, 'y'))
                self.logloss_mean(self.mymean / np.mean(self.mymean) / 4.0, self.getdataNN('keihintohoku', i, 'y'))

    def logloss(self,pred,act):
        tmp = np.sum(pred * act)
        if tmp == 0:
            tmp = 1.0e-15
        self.loss_sum += -1*np.log(tmp)
    def logloss_mean(self,pred,act):
        tmp=np.sum(pred * act)
        if tmp == 0:
            tmp=1.0e-15
        self.loss_mean += -1*np.log(tmp)


if __name__=='__main__':
    my=Validation(20)
    my.run()

