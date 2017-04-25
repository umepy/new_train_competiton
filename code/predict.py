#coding:utf-8
#予測して提出用ファイルを作成

import numpy as np
np.random.seed(114514)
import pickle
import sys
sys.path.append('./code')
from lstm import lstmC
from gboost import gboost
import tqdm


class predict():
    def __init__(self):
        self.distance=20
        self.name = ['keihintohoku', 'keiyou', 'saikyoukawagoe', 'tyuou', 'uchibou']
        self.tname = ['sotobou', 'syonan', 'takasaki', 'utsunomiya', 'yamanote']
        self.read_data()
    def getalllstmdata(self, name, num):
        data_x = self.Time_x[name[0]][:num]
        data_y = self.Time_y[name[0]][:num]
        for i in range(1, len(name)):
            data_x = np.concatenate((data_x, self.Time_x[name[i]][:num]), axis=0)
            data_y = np.concatenate((data_y, self.Time_y[name[i]][:num]), axis=0)
        return data_x, data_y
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
            with open('../data/pickle/18' + i + '_' + str(self.distance) + 'km_x.pickle', 'rb') as f:
                self.Time_x[i] = pickle.load(f)
            with open('../data/pickle/18' + i + '_' + str(self.distance) + 'km_y.pickle', 'rb') as f:
                self.Time_y[i] = pickle.load(f)
        for i in self.tname:
            with open('../data/pickle/18' + i + '_' + str(self.distance) + 'km_x.pickle', 'rb') as f:
                self.test[i] = pickle.load(f)
    def run(self):
        model=lstmC()
        flag=0
        result={}
        lastnum=0
        for i in self.tname:
            result[i]=np.array([[0,0,0,0]])
        #10000から始める 18考慮の9982
        for i in tqdm.tqdm(range(9982,len(self.test['sotobou']))):
            #たいむすぱんを18とする
            #初期学習
            if (i+18)%1000==0 and flag==0:
                x,y=self.getalllstmdata(self.name,i)
                model.fit(x,y,512,10)
                flag=1
            #ここから1000個前までの学習を使って予測をしていく
            #その後学習
            elif (i+18)%1000==0 and flag==1:
                for j in self.tname:
                    result[j]=np.concatenate((result[j],model.predict(self.test[j][i-999:i+1])))
                model = lstmC()
                x, y = self.getalllstmdata(self.name, i)
                model.fit(x, y, 512, 10)
                lastnum=i
            if (i+18+1000)>len(self.test['sotobou']):
                print(i)
                for j in self.tname:
                    result[j]=np.concatenate((result[j],model.predict(self.test[j][lastnum + 1:])))
                break
        with open('../result/lstm_predict.pickle','wb') as f:
            pickle.dump(result,f)

if __name__=='__main__':
    my=predict()
    my.run()
