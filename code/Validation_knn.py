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
from lstm import lstmC
import tqdm
from sklearn.metrics import log_loss


class Validation():
    def __init__(self,dis):
        self.set_distance(dis)
        self.read_data()
        self.loss_sum=[0,0,0,0,0]
        self.loss_mean=[0,0,0,0,0]
        self.loss_lstm = [0, 0, 0, 0, 0]
        self.name = ['keihintohoku','keiyou','saikyoukawagoe','tyuou','uchibou']
        #number
        self.mymean=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
    def set_distance(self,dis):
        self.distance = dis
    def read_data(self):
        self.trains={}
        self.NNtrains = {}
        self.Time_x={}
        self.Time_y = {}
        for i in ['keihintohoku','keiyou','saikyoukawagoe','tyuou','uchibou']:
            with open('../data/pickle/'+i+'_'+str(self.distance)+'.pickle','rb') as f:
                self.trains[i] = pickle.load(f).as_matrix()
            with open('../data/pickle/'+i+'_'+str(self.distance)+'_NN.pickle','rb') as f:
                self.NNtrains[i] = pickle.load(f).as_matrix()
            with open('../data/pickle/18'+i+'_'+str(self.distance)+'km_x.pickle','rb') as f:
                self.Time_x[i] = pickle.load(f)
            with open('../data/pickle/18' + i + '_' + str(self.distance) + 'km_y.pickle', 'rb') as f:
                self.Time_y[i] = pickle.load(f)
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
        if type(name) != type([]):
            return self.trains[name][0:num, 1:], None
        data = self.trains[name[0]]
        for i in range(1, len(name)):
            data = np.concatenate((data, self.trains[name[i]]), axis=0)

        #学習用のダミー訓練データを挿入
        x_data = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
        y_data = np.array([0,1,2,3])
        x_data = np.concatenate((x_data,data[0:num*4,1:]))
        y_data = np.concatenate((y_data,data[4:(num+1)*4,0]))
        return x_data,y_data
    def getalldataNN(self,name,num):
        data = self.NNtrains[name]
        return data[1:num+1,:4]
    def getalllstmdata(self,name,num):
        data_x = self.Time_x[name[0]][:num]
        data_y = self.Time_y[name[0]][:num]
        for i in range(1, len(name)):
            data_x = np.concatenate((data_x, self.Time_x[name[i]][:num]), axis=0)
            data_y = np.concatenate((data_y, self.Time_y[name[i]][:num]), axis=0)
        return data_x,data_y
    def getlstmdata(self,name,num):
        data_x = self.Time_x[name][:num+1]
        return data_x
    def logloss(self,pred,act,j):
            self.loss_sum[j] += log_loss(act,pred,normalize=False)
    def logloss_mean(self,pred,act,j):
        tmp=[pred]*len(act)
        self.loss_mean[j] += log_loss(act,tmp,normalize=False)
    def logloss_lstm(self,pred,act,j):
        kotei = np.array([self.mymean[j] / np.mean(self.mymean[j]) / 4.0])
        kotei_2=np.array([self.mymean[j]/np.mean(self.mymean[j])/4.0])
        for _ in range(len(act) - len(pred)-1):
            kotei=np.concatenate((kotei,kotei_2),axis=0)
        pred=np.concatenate((kotei,np.array(pred)))
        self.loss_lstm[j] += log_loss(act,pred,normalize=False)
    def run(self):
        #５通りの訓練データを作成する
        train_names=[]
        train_names.append(['keiyou','saikyoukawagoe','tyuou','uchibou'])
        train_names.append(['keihintohoku','saikyoukawagoe','tyuou','uchibou'])
        train_names.append(['keihintohoku','keiyou','tyuou','uchibou'])
        train_names.append(['keihintohoku','keiyou','saikyoukawagoe','uchibou'])
        train_names.append(['keihintohoku','keiyou','saikyoukawagoe','tyuou'])
        models = []
        nn_models=[]
        for i in range(5):
            models.append(gboost(True))
            nn_models.append(lstmC())
        result=[]
        for i in range(1,12000):
            if i%1000==0:
                nn_models = []
                for _ in range(5):
                    nn_models.append(lstmC())
            for j in range(5):
                for nn in train_names[j]:
                    self.mymean[j][np.argmax(self.getdataNN(nn, i - 1, 'y'))] += 1

                if i%5==0:
                    x, y = self.getalldata(train_names[j], i)
                    models[j].fit(x,y)
                if i%1000==0 and i>16:
                    #時系列入力のためずらす
                    x,y = self.getalllstmdata(train_names[j],i-16)
                    nn_models[j].fit(x,y,512,10)

                #誤差計算
                if i%1000==0:
                    x, y = self.getalldata(self.name[j], i)
                    self.logloss(models[j].predict(x),self.getalldataNN(self.name[j],i),j)
                    self.logloss_mean(self.mymean[j]/np.mean(self.mymean[j])/4.0, self.getalldataNN(self.name[j],i),j)
                    self.logloss_lstm(nn_models[j].predict(self.getlstmdata(self.name[j],i-16)),self.getalldataNN(self.name[j],i),j)
            #誤差集計＆表示
            if i%1000==0:
                print(str(i) + ':\t' + str(np.mean(self.loss_sum) / i) + '\t' + str(np.mean(self.loss_mean) / i) + '\t'+str(np.mean(self.loss_lstm)/i))
                result.append([np.mean(self.loss_sum) / i,np.mean(self.loss_mean) / i,np.mean(self.loss_lstm)/i])
                self.loss_sum=[0,0,0,0,0]
                self.loss_mean=[0,0,0,0,0]
                self.loss_lstm=[0,0,0,0,0]
        with open('result/val_result.pickle','wb') as f:
            pickle.dump(result,f)

if __name__=='__main__':
    my=Validation(20)
    my.run()

