
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import normalize
import networkx as nx
from collections import defaultdict
from utils import load_data

from sklearn.model_selection import train_test_split

from sklearn import svm

models = ['DistMult','TransE','HolE','ComplEx','RotatE','pRotatE','HAKE','ConvE','ConvKB']

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def explained_variance():
    # explained variance of embeddings
    mat = []
    d = './results/pretrained_embeddings/'
    labels = [[]]

    
    for model1, model2 in product(models,models):
        X,y = load_data('./data/data.csv')
        y = np.asarray(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)

        f = d+model1+'_chemical_embeddings.npy'
        X1 = np.load(f)
        #X1 = normalize(X1,norm='l2',axis=1) #normalize each vector
        f = d+model1+'_chemical_ids.npy'
        ids1 = dict(np.load(f))

        f = d+model2+'_taxonomy_embeddings.npy'
        X2 = np.load(f)
        #X2 = normalize(X2,norm='l2',axis=1)
        f = d+model2+'_taxonomy_ids.npy'
        ids2 = dict(np.load(f))
    
        X = np.asarray([np.concatenate([X1[int(ids1[c])],X2[int(ids2[s])],[conc]],axis=0) for c,s,conc in X_train if c in ids1 and s in ids2])
        #X = normalize(X,norm='l2',axis=1) # normalize over each feature
        
        nc = 1
        exp_var = 0
        while exp_var < 0.9 and nc < len(X[0]):
            nc += 1
            pca = PCA(n_components=nc)
            X_prime = pca.fit_transform(X)
            exp_var = sum(pca.explained_variance_ratio_)
            
        mat.append(nc/len(X[0]))
    
        pca = PCA(n_components=2)
        X_prime = pca.fit_transform(X)
        rgba_colors = np.zeros((len(X),4))
        # for red the first column needs to be one
        rgba_colors[np.where(y_train>0.5),0] = 1.0
        rgba_colors[np.where(y_train<=0.5),2] = 1.0
        
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = 1.0
        fig, ax = plt.subplots()
        im = ax.scatter(x=X_prime[:,0],y=X_prime[:,1],color=rgba_colors)
        ax.set_xlabel('Principal component 1', fontsize=18)
        ax.set_ylabel('Principal component 2', fontsize=18)
        fig.tight_layout()
        fig.set_size_inches(10, 10)
        plt.savefig('./plots/pca_2d_'+model1+'_'+model2+'.png')
        plt.close()
        
        labels[i][j] = str
        
    mat = np.reshape(np.asarray(mat),(len(models),len(models)))
    mat = np.around(mat,2)
    fig, ax = plt.subplots()
    im = ax.imshow(mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, labels[i][j],
                        ha="center", va="center", color="w")

    #ax.set_title("Number of PCA components for explained variance over 0.9.")
    ax.set_xlabel('Chemical KG', fontsize=18)
    ax.set_ylabel('Taxonomy KG', fontsize=18)
    fig.tight_layout()
    fig.set_size_inches(10, 10)
    plt.savefig('./plots/pca_components.png')
    plt.close()
    
def reduce_kg_to_hier(kg,hier_rels):
    tmp = []
    for s,p,o in kg:
        for direction,rel in hier_rels:
            if p == rel:
                if direction > 0:
                    tmp.append((s,o))
                else:
                    tmp.append((o,s))
    return tmp

def reduce_kg_to_objects(kg,objects,type_rel):
    tmp = set()
    for s,p,o in kg:
        if p == type_rel and o in objects:
            tmp.add(s)
    return [(s,p,o) for s,p,o in kg if s in tmp and o in tmp]

def to_networkx_graph(edges):
    g = nx.DiGraph()
    for n1,n2 in edges:
        g.add_edge(n1,n2)
    return g

def calculate_score(g,embeddings,ids):
    leafs = [x for x in g.nodes() if g.in_degree(x)==0]
    score = 0
    d = defaultdict(set)
    for l in leafs:
        if l in g:
            for s in g.successors(l):
                d[s].add(l)
    
    transforms = []
    
    for k in d:
        if len(d[k]) > 1:
            X = np.mean(np.asarray([embeddings[int(ids[n])] for n in d[k]]),axis=0)
            M = embeddings[int(ids[k])]
            T = np.linalg.solve(np.diag(X), M)
            transforms.append(T)
        for n in d[k]:
            if n in g: g.remove_node(n)
    
    for T1, T2 in product(transforms,transforms):
        score += np.mean(abs(T1-T2))
    score = score/max(1,len(transforms))**2
    
    if len(d.keys()) > 0:
        score += calculate_score(g,embeddings,ids)
    
    return score
    
def locality():
    
    kg1 = pd.read_csv('./chemicals0.csv')
    kg2 = pd.read_csv('./taxonomy0.csv')
    
    kg1 = list(zip(kg1['subject'], kg1['predicate'], kg1['object']))
    kg2 = list(zip(kg2['subject'], kg2['predicate'], kg2['object']))
    
    type_rel = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    
    kg1_hier_rels = [(1,'http://id.nlm.nih.gov/mesh/vocab#broaderConcept'),
                     (-1,'http://id.nlm.nih.gov/mesh/vocab#narrowerConcept')]
    kg1_hier_objects = ['http://id.nlm.nih.gov/mesh/vocab#Concept']
    
    kg2_hier_rels = [(1,'http://www.w3.org/2000/01/rdf-schema#subClassOf')]
    kg2_hier_objects = ['https://www.ncbi.nlm.nih.gov/taxonomy/Taxon']
    
    kg1 = reduce_kg_to_objects(kg1,kg1_hier_objects,type_rel)
    kg2 = reduce_kg_to_objects(kg2,kg2_hier_objects,type_rel)
    kg1 = reduce_kg_to_hier(kg1,kg1_hier_rels)
    kg2 = reduce_kg_to_hier(kg2,kg2_hier_rels)
    
    tmp1 = []
    
    for model in models:
        g1 = to_networkx_graph(kg1)
        f = 'pretrained/'+model+'_chemical_embeddings.npy'
        X = np.load(f)
        X = normalize(X,norm='l2',axis=1)
        f = 'pretrained/'+model+'_chemical_ids.npy'
        ids = dict(np.load(f))
        tmp1.append(calculate_score(g1,X,ids))
            
    tmp2 = []
    
    for model in models:
        g2 = to_networkx_graph(kg2)
    
        f = 'pretrained/'+model+'_taxonomy_embeddings.npy'
        X = np.load(f)
        X = normalize(X,norm='l2',axis=1)
        f = 'pretrained/'+model+'_taxonomy_ids.npy'
        ids = dict(np.load(f))
        tmp2.append(calculate_score(g2,X,ids))
    
    mat = []
    for a,b in product(tmp1,tmp2):
        mat.append(a*b)
    
    mat = np.reshape(np.asarray(mat),(len(models),len(models)))
    mat = np.around(mat,2)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    im = ax.imshow(mat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, mat[i, j],
                        ha="center", va="center", color="w")

    #ax.set_title("Locality metric")
    ax.set_xlabel('Chemical KG', fontsize=18)
    ax.set_ylabel('Taxonomy KG', fontsize=18)
    fig.tight_layout()
    plt.savefig('./plots/locality.png')
    plt.close()


if __name__ == '__main__':
    explained_variance()
    #locality()
    
    
    
    
    
    
