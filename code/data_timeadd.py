#coding:utf-8
#時間の説明変数を追加するスクリプト

import pandas as pd
import pickle
import pprint
import numpy as np
import sys
import jholiday
sys.path.append('./code')

#休日と土日に1を返す関数
def doniti(date):
    if date.weekday()>=5:
        return 1
    else:
        if jholiday.holiday_name(year=date.year,month=date.month,day=date.day) != None:
            return 1
        else:
            return 0

class timeadd():
    def __init__(self,name,distance=20):
        self.train = pd.read_csv('../data/points/'+name+'_'+str(distance)+'_train.csv')
        self.timedf=pd.read_csv('../data/train.csv',parse_dates=True,index_col=0)
        self.timedf = self.timedf.index.tolist()
        #時間をcos,sinに変換
        self.cos = list(map(lambda x: np.cos(x.hour/24.0 * np.pi *2),self.timedf))
        self.sin = list(map(lambda x: np.sin(x.hour / 24.0 * np.pi * 2), self.timedf))

        self.doniti = list(map(lambda x: doniti(x),self.timedf))

        self.train['hour_cos'] = self.cos
        self.train['hour_sin'] = self.sin
        self.train['holiday'] = self.doniti
        df = pd.DataFrame(pd.get_dummies(self.train['state']).as_matrix(), columns=['none', 'people', 'machine', 'weather'])
        for i in self.train.columns:
            if i!='state':
                df[i] = self.train[i].as_matrix()

        pprint.pprint(df)
        with open('../data/pickle/'+name+'_'+str(distance)+'.pickle', 'wb') as f:
            pickle.dump(df,f)

if __name__=='__main__':
    my=timeadd('keihintohoku',20)