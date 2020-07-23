# analyse results

import pandas as pd

kge_models = ['DistMult']

sampling_methods = ['none','chemical','species','both']

d = './results/'
k = 10
import glob

m = 'auc'

for sm in sampling_methods:
    files = glob.glob(d+sm+'*.csv')

    best_auc = [0]*k
    best_model = ['']*k
    for f in files:
        df = pd.read_csv(f,index_col='metric')
        if df.loc[m,'value'] > min(best_auc):
            idx = best_auc.index(min(best_auc))
            best_auc[idx] = df.loc[m,'value']
            best_model[idx] = f
            
    print(sm)
    for ba,bm in zip(best_auc,best_model):
        print(ba,bm)
    print('\n')
