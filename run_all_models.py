

#from models.utils import load_data, train_test_split_custom

#from models.sim_embedding_models import fit_sim_model

from models.pretrained_auto_keras import fit_onehot, fit_pretrained, fit_hier_embeddings, fit_hier_kg_combination

import numpy as np
import pandas as pd

from sklearn.utils import class_weight
import glob

DATA_FILE = './data/cocoa_data.csv'
NUM_RUNS = 1
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
MAX_EPOCHS = 100
SEARCH_MAX_EPOCHS = 10
PATIENCE = 10
NUM_FOLDS = 5
layers_1 = [400 ,200, 100]
layers_2 = []
loss_func = 'categorical_crossentropy'
output_dim = 41

PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE,
        'SEARCH_MAX_EPOCHS':SEARCH_MAX_EPOCHS,
        'NUM_FOLDS':NUM_FOLDS
    }

hier_embeddings_dir = '../hierarchy-based-embeddings/embeddings/'

KGE_EMBEDDINGS_DIR = './results/pretrained_embeddings/'

#chemical_hier_embeddings_files = file_list = glob.glob(hier_embeddings_dir+'*_hierarchy.csv') 0.5288300126791
#chemical_hier_embeddings_files = file_list = glob.glob(hier_embeddings_dir+'*_forest.csv') #0.5435945630073548

chemical_hier_embeddings_files = [hier_embeddings_dir + 'country_embeddings.csv']

y='BeanType'

def main():
    
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=[['BeanType']])
    y = data['BeanType'].values
    entities1,  _ = map(set,zip(*X))
    me1 = {k:i for i,k in enumerate(entities1)}
    
    print('Prior',np.unique(y,return_counts=True)[1]/sum(np.unique(y,return_counts=True)[1]))
    
    #for SAMPLING in ['none','chemical','species','both']:
    for SAMPLING in ['none']:
        
        #X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.20, sampling=SAMPLING)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, sampling=SAMPLING)
        
        PARAMS['cw'] = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train))) 
        
        # test one-hot model
        #fit_onehot(X_train, X_test, y_train, y_test, me1, me2,
             #      results_file='./results/%s_one_hot.csv' % SAMPLING, 
              #     hp_file = './pred_hp/%s_one_hot.csv' % SAMPLING,
               #    num_runs = NUM_RUNS,
                #   params=PARAMS)
        
        # test hier embeddings 
        fit_hier_embeddings(X_train, X_test, y_train, y_test, layers_1, layers_2, loss_func, output_dim,
                            hier_embeddings_files,
                            results_file='./results/%s_hierarchy_embedding.csv' % SAMPLING,
                            hp_file='./pred_hp/%s_hierarchy_embedding.csv' % SAMPLING,
                            num_runs = NUM_RUNS,
                            params=PARAMS)
        
 
                             
    
if __name__ == '__main__':
    main()
    
    
    
