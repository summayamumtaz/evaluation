

#from models.utils import load_data, train_test_split_custom

#from models.sim_embedding_models import fit_sim_model

import numpy as np
import pandas as pd
import tensorflow as tf
import kerastuner as kt
import kerastuner
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner import HyperParameters, Objective
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate
#from tensorflow_addons.callbacks import TimeStopping
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy, Metric
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import preprocessing


from sklearn.utils import class_weight
import glob

DATA_FILE = './data/cocoa_data.csv'
NUM_RUNS = 1
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
MAX_EPOCHS = 100
SEARCH_MAX_EPOCHS = 10
PATIENCE = 10
units= [32,16,8]
loss_func = 'categorical_crossentropy'
output_dim = 17
EXECUTIONS_PER_TRIAL =10
dim1 = 2
dim2 = 47

def tune_optimizer_model(hp, dim1, dim2):
    
    ci = Input((dim1,))
    si = Input((dim2,))
    s=si
    
    for i in range(hp.Int('num_layers', 0, 2)):
        s = Dense(hp.Choice('branching_units'+str(i+1),units,default=units[0]),activation='relu')(s)
        s = Dropout(0.2)(s)
        
    x = Concatenate(axis=-1)([ci,s])
    
    x1 = Dense(hp.Choice('units_'+str(1),[16,8], default=16), activation='relu')(x)
   
        
    x = Dense(output_dim,activation='softmax',name='output_1')(x1)
    
    model = Model(inputs=[ci,si],outputs=[x])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'output_1':'categorical_crossentropy'},
                  metrics = ['accuracy'])
    
    return model

def fit_one_hot(X,y, result_dir, project):
    X_onehot = pd.get_dummies(X['BeanOrigin'], drop_first=True)
    X = pd.concat( [X[['Rating', 'CocoaPercent']],X_onehot], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_train = to_categorical(y_train, output_dim)
    y_test = to_categorical(y_test, output_dim)
    
    X_train1= X_train[['Rating', 'CocoaPercent']].values
    X_train2 = X_train.drop([ 'Rating', 'CocoaPercent'],axis=1).values
    X_test1= X_test[['Rating', 'CocoaPercent']].values
    X_test2 = X_test.drop(['Rating', 'CocoaPercent'],axis=1).values

    dim1 = X_train1.shape[1]
    dim2 = X_train2.shape[1]
    print (dim1, dim2)
    
    dim1 = X_train1.shape[1]
    dim2 = X_train2.shape[1]
    
    hp = HyperParameters()
    
    bm = lambda x: tune_optimizer_model(hp, dim1, dim2)
   
    tuner = RandomSearch(
        bm,
        objective='val_accuracy',
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=result_dir, 
        project_name=project,
        seed=32)

    TRAIN_EPOCHS = 1000

    tuner.search(x=[X_train1,X_train2],
                 y=y_train,
                 epochs=TRAIN_EPOCHS,
                 validation_data=([X_test1,X_test2], y_test))
    tuner.results_summary()



def fit_hier_embedding(X,y, result_dir, project):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_train = to_categorical(y_train, output_dim)
    y_test = to_categorical(y_test, output_dim)
    
    X_train1= X_train[['Rating', 'CocoaPercent']].values
    X_train2 = X_train.drop([ 'Rating', 'CocoaPercent'],axis=1).values
    X_test1= X_test[['Rating', 'CocoaPercent']].values
    X_test2 = X_test.drop(['Rating', 'CocoaPercent'],axis=1).values

    dim1 = X_train1.shape[1]
    dim2 = X_train2.shape[1]
    
    hp = HyperParameters()
    
    bm = lambda x: tune_optimizer_model(hp, dim1, dim2)
    
    print(dim1, dim2)
    tuner = RandomSearch(
        bm,
        objective='val_accuracy',
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=result_dir, 
        project_name=project,
        seed=32)

    TRAIN_EPOCHS = 1000

    tuner.search(x=[X_train1,X_train2],
                 y=y_train,
                 epochs=TRAIN_EPOCHS,
                 validation_data=([X_test1,X_test2], y_test))
    tuner.results_summary()








def main():
    
    data = pd.read_csv(DATA_FILE)
    X_onehot = data[['BeanOrigin','Rating', 'CocoaPercent' ]]
    y = data['BeanType'].values
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    #fit_one_hot(X,y,'result','cocoa_onehot')
    
    X_embed = data.drop(['BeanType', 'BeanOrigin'], axis=1)
    fit_hier_embedding(X_embed,y, 'result', 'cocoa_Hierarchy')



                             
    
if __name__ == '__main__':
    main()
    
    
    
