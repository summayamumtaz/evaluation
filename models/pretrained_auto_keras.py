## Use pretrained embeddings to train a auto-sklearn classisifier.

import numpy as np
import pandas as pd
import tensorflow as tf
import kerastuner as kt

from kerastuner import HyperParameters, Objective

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate
#from tensorflow_addons.callbacks import TimeStopping
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy, Metric
from tensorflow.keras.optimizers import Adam

from .utils import f1, f2, CVTuner, reset_weights, create_class_weight

import json
import random
import string

MAX_EPOCHS = 1000
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
PATIENCE = 5
PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE
    }


         
def build_model(hp, input_dim1, input_dim2, output_dim,layers_1, layers_2,loss_func='Binary_crossentropy' ,first_layer=False):
    
    params = hp.values.copy()
    
    ci = Input((input_dim1,))
    si = Input((input_dim2,))
    
    
    if first_layer:
        c = Embedding(params['num_entities1'],hp.Choice('embedding_dim1',[100,200]))(ci)
        s = Embedding(params['num_entities2'],hp.Choice('embedding_dim2',[100,200]))(si)
        c = tf.squeeze(c,axis=1)
        s = tf.squeeze(s,axis=1)
    else:
        c = ci
        s = si
    conc = conci
        
    for i,layer_num in enumerate(range(hp.Int('branching_num_layers_chemical',0,len(layers_1),default=1))):
        c = Dense(hp.Choice('branching_units_chemical_'+str(i+1),layers_1,default=layers_1[0]),activation='relu')(c)
        c = Dropout(0.2)(c)
    
    if layers_2:
        for i,layer_num in enumerate(range(hp.Int('branching_num_layers_species',0,len(layers_2),default=1))):
            s = Dense(hp.Choice('branching_units_species_'+str(i+1),layers_2,default=layers_2[0]),activation='relu')(s)
            s = Dropout(0.2)(s)
        
    
    x = Concatenate(axis=-1)([c,s])
    
    for i,layer_num in enumerate(range(hp.Int('num_layers',0,3,default=1))):
        x = Dense(hp.Choice('units_'+str(i+1),[32,128,512], default=128), activation='relu')(x)
        x = Dropout(0.2)(x)
        
    x = Dense(output_dim ,activation='softmax',name='output_1')(x)
    
    model = Model(inputs=[ci,si,conci],outputs=[x])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'output_1': loss_func},
                  metrics=['acc', f1, f2, Precision(), Recall(), AUC()])
    
    return model

def tune(X_train, X_test, y_train, y_test, 
         hypermodel,
         hp,
         params,
         num_runs,
         results_file,
         hp_file):
    
    tuner = CVTuner(
        hypermodel=hypermodel,
        oracle=kt.oracles.BayesianOptimization(
            hyperparameters=hp,
            objective=Objective('val_auc','max'),
            max_trials=params['MAX_TRIALS']),
        overwrite=True,
        project_name='tmp/'+''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(11))
        )

    tuner.search(X_train,y_train,
        epochs=params['MAX_EPOCHS'],
        batch_size=1024,
        callbacks=[EarlyStopping('loss',mode='min',patience=params['PATIENCE']),
                   ReduceLROnPlateau('loss',mode='min',patience=params['PATIENCE'])],
        class_weight = params['cw'],
        kfolds=params['NUM_FOLDS']
        )

    results = []
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)
    for _ in range(num_runs):
        model.fit(X_train, y_train,
                epochs=params['MAX_EPOCHS'],
                batch_size=1024,
                class_weight = params['cw'],
                callbacks=[EarlyStopping('loss',mode='min',patience=params['PATIENCE']),
                           ReduceLROnPlateau('loss',mode='min',patience=params['PATIENCE'])],
                verbose=0)
        r = model.evaluate(X_test,y_test,verbose=0)
        results.append(r)
        reset_weights(model)
            
    var = np.var(np.asarray(results),axis=0)
    results = np.mean(np.asarray(results),axis=0)
    
    df = pd.DataFrame(data={'metric':model.metrics_names,'value':list(results), 'variance':list(var)})
    df.to_csv(results_file)
    
    out = dict()
    for k in best_hps.values.keys():
        out[k] = best_hps.values[k]
    with open(hp_file, 'w') as fp:
        json.dump(out, fp)

class PriorModel:
    def __init__(self):
        pass
    def fit(self,X,y):
        u,uw = np.unique(y,return_counts=True)
        self.lookup = uw/sum(uw)
    
    def predict(self,X):
        return np.asarray([np.argmax(self.lookup) for _ in range(len(X))])

def fit_onehot(X_train, X_test, y_train, y_test, me1, me2, results_file='results.csv',hp_file='hp.json',num_runs=1, params=None):
    #one hot
    params = params or PARAMS
    
    X_train,y_train = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_train]), np.asarray(y_train)
    X_test,y_test = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_test]), np.asarray(y_test)
    
    X_train = [np.asarray(a) for a in zip(*X_train)]
    X_test = [np.asarray(a) for a in zip(*X_test)]
    
    hp = HyperParameters()
    hp.Fixed('num_entities1',len(me1))
    hp.Fixed('num_entities2',len(me2))
    
    bm = lambda x: build_model(x,1,1,first_layer=True)
    tune(X_train, X_test, y_train, y_test, 
         bm,
         hp,
         params,
         num_runs,
         results_file,
         hp_file)


        
def load_hier_embeddings(f,entities):
    X = np.diag(np.zeros(len(entities)))
    df = pd.read_csv(f)
    cols = list(df.columns[1:-1])
    for c1 in cols:
        try:
            i = entities.index(c1)
        except:
            pass
        for c2 in cols:
            try:
                j = entities.index(c2)
                X[i,j] = min(1,df.iloc[cols.index(c1)+1,cols.index(c2)+1])
            except:
                pass
    return X

def create_hier_data(X_train, X_test, y_train, y_test, chemical_embedding_files):
    entities11,  _ = map(set,zip(*X_train))
    entities12,  _ = map(set,zip(*X_test))
    entities1 = entities11 | entities12
    
    
    entities1 = list(entities1)
    
    
    X1 = []
    for f in chemical_embedding_files:
        X1.append(load_hier_embeddings(f,entities1))
    X1 = np.concatenate(X1,axis=1)
    
    
    
    me1 = {k:i for i,k in enumerate(entities1)}
    rme1 = {i:k for k,i in me1.items()}
    
    
    X_train = np.asarray([[X1[me1[a],:],[float(c)]] for a,b,c in X_train])
    X_test = np.asarray([[X1[me1[a],:],[float(c)]] for a,b,c in X_test])
    
    X_train = [np.asarray(a) for a in zip(*X_train)]
    X_test = [np.asarray(a) for a in zip(*X_test)]
    return X_train, X_test, y_train, y_test



def fit_hier_embeddings(X_train, X_test, y_train, y_test, layers_1, layers_2, loss_func, output_dim, chemical_embedding_files, results_file='results.csv',hp_file='hp.json', num_runs=1, params=None):
    params = params or PARAMS

    X_train, X_test, y_train, y_test = create_hier_data(X_train, X_test, y_train, y_test,chemical_embedding_files)
    
    y_train = np.asarray(y_train).reshape((-1,1))
    y_test = np.asarray(y_test).reshape((-1,1))
    
    hp = HyperParameters()
    
    bm = lambda x: build_model(x,len(X_train[0][0]),len(X_train[1][0],layers_1, layers_2, loss_func , output_dim ))
    tune(X_train, X_test, y_train, y_test, 
         bm,
         hp,
         params,
         num_runs,
         results_file,
         hp_file)
    
