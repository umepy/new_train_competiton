#coding:utf-8
#gboostで予測する

import pickle
import numpy as np
np.random.seed(114514)
import sys
sys.path.append('./code')
from gboost import gboost
import tqdm

class predict():
    def __init__(self):
        self.distance=20
        self.name = ['keihintohoku', 'keiyou', 'saikyoukawagoe', 'tyuou', 'uchibou']
        self.tname = ['sotobou', 'syonan', 'takasaki', 'utsunomiya', 'yamanote']
        self.read_data()
    def getalldata(self, name, num):
        data_x = self.trains[name[0]][:num,1:]
        data_y = self.trains[name[0]][1:num+1,0]
        for i in range(1, len(name)):
            data_x = np.concatenate((data_x, self.trains[name[i]][:num,1:]), axis=0)
            data_y = np.concatenate((data_y, self.trains[name[i]][1:num+1,0]), axis=0)
        # 学習用のダミー訓練データを挿入
        x_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]])
        y_data = np.array([0, 1, 2, 3])
        x_data = np.concatenate((x_data, data_x))
        y_data = np.concatenate((y_data, data_y))
        return x_data, y_data
    def read_data(self):
        self.trains = {}
        self.NNtrains = {}
        self.Time_x = {}
        self.Time_y = {}
        self.test = {}
        for i in ['keihintohoku', 'keiyou', 'saikyoukawagoe', 'tyuou', 'uchibou']:
            with open('../data/pickle/' + i + '_' + str(self.distance) + '.pickle', 'rb') as f:
                self.trains[i] = pickle.load(f).as_matrix()
            with open('../data/pickle/' + i + '_' + str(self.distance) + '_NN.pickle', 'rb') as f:
                self.NNtrains[i] = pickle.load(f).as_matrix()
        for i in self.tname:
            with open('../data/pickle/' + i + '_' + str(self.distance) + '.pickle', 'rb') as f:
                self.test[i] = pickle.load(f)
    def run(self):
        model=gboost(False)
        result={}
        for i in self.tname:
            result[i]=np.array([[0,0,0,0]])
        #10000から始める 18考慮の9982
        model = gboost(False)
        interval=0
        for i in tqdm.tqdm(range(1,10001)):
            if i>1000:
                interval=5
            if i>5000:
                interval=10
            if i%interval==0:
                x,y=self.getalldata(self.name,i)
                model.fit(x,y)
            for j in self.tname:
                result[j]=np.concatenate((result[j],model.predict(self.test[j][i-1:i])))
        with open('../result/gboost_predict.pickle','wb') as f:
            pickle.dump(result,f)

if __name__=='__main__':
    my=predict()
    my.run()