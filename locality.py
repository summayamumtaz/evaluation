### Locallity

import numpy as np
from itertools import product
from collections import defaultdict
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(self, name, vector):
        
        self.parent = None
        self.children = []
        self.name = name
        self.vector = vector

    def add_parent(self, p):
        self.parent = p
    def add_child(self, c):
        self.children.append(c)

    def __str__(self):
        return str(self.name)
    
    def __hash__(self):
        return self.name
    
    def __eq__(self, other):
        return self.__str__() == other.__str__()

def find_transform(x,x_p):
    # solve x*T = x_p
    T = np.linalg.solve(np.diag(x), x_p)
    return T

def diff_transform(M,N):
    return np.mean(abs(abs(M)-abs(N)))

def generate_score(nodes):
    
    score = 0
    
    transforms = []
    clusters = defaultdict(list)
    for n in nodes:
        if n.parent != None:
            clusters[n.parent].append(n)
            
    for k in clusters:
        X = np.mean(np.asarray([a.vector for a in clusters[k]]),axis=0)
        transforms.append(find_transform(X,k.vector))
    
    for T1,T2 in product(transforms, transforms):
        score += diff_transform(T1,T2)
    
    #if clusters:
        #score += generate_score(clusters.keys())
    
    return score

def test_cluster_mean():
    return all(all(l) for l in np.ones((10,1)) == cluster_center(np.ones((10,10))))

def test_find_transform():
    x = np.ones((10,))
    x_p = 2*np.ones((10,))
    T = find_transform(x,x_p)
    return all(2*np.diag(np.ones((10,1))) == T)

def test_diff_transform():
    X = np.ones((10,10))
    return diff_transform(X,X) == 0
    
def create_nodes(edges, embeddings):
    nodes = []
    for s,o in edges:
        if not s in nodes:
            n1 = Node(s,embeddings[s-1])
        else:
            n1 = nodes[nodes.index(s)]
        if not o in nodes:
            n2 = Node(o,embeddings[o-1])
        else:
            n2 = nodes[nodes.index(o)]
            
        n2.add_child(n1)
        n1.add_parent(n2)
        nodes.append(n1)
        nodes.append(n2)
    return nodes
    
def test():
    
    #assert test_cluster_mean()
    #assert test_find_transform()
    #assert test_diff_transform()
    
    edges = [[1,6],[2,6],[3,5],[4,5],[6,7],[5,7]] # subject,predicate,object
    
    embeddings = np.asarray([
            [1,1],
            [2,1],
            [4,1],
            [5,1],
            [4.5,2],
            [1.5,2],
            [3,3],
        ])
    
    nodes = create_nodes(edges, embeddings)
    leaf_nodes = [n for n in nodes if len(n.children) < 1]
    score = generate_score(leaf_nodes)
    assert score == 0.0

def main():
    kg1 = pd.read_csv('./chemicals.csv')
    kg2 = pd.read_csv('./taxonomy.csv')
    
    kg1 = list(zip(kg1['subject'], kg1['predicate'], kg1['object']))
    kg2 = list(zip(kg2['subject'], kg2['predicate'], kg2['object']))
    
    g1 = nx.DiGraph()
    for s,p,o in kg1:
        if p == 'http://id.nlm.nih.gov/mesh/vocab#narrowerConcept':
            g1.add_edge(o,s)
        if p == 'http://id.nlm.nih.gov/mesh/vocab#broaderConcept':
            g1.add_edge(s,o)
    
    taxons = set([s for s,p,o in kg2 if p=='http://www.w3.org/1999/02/22-rdf-syntax-ns#type' and o=='https://www.ncbi.nlm.nih.gov/taxonomy/Taxon'])
    species = set([s for s,p,o in kg2 if p=='https://www.ncbi.nlm.nih.gov/taxonomy/rank' and o=='https://www.ncbi.nlm.nih.gov/taxonomy/rank/species'])
    
    kg2 = [(s,p,o) for s,p,o in kg2 if s in taxons and o in taxons]
    g2 = nx.DiGraph()
    for s,p,o in kg2:
        if p == 'http://www.w3.org/2000/01/rdf-schema#subClassOf':
            g2.add_edge(s,o)
    
    #L = nx.linalg.laplacianmatrix.directed_laplacian_matrix(g2)
    
    exit()
    
    for s,p,o in kg1:
        if p == 'http://id.nlm.nih.gov/mesh/vocab#narrowerConcept':
            kg1.append((o,'http://id.nlm.nih.gov/mesh/vocab#broaderConcept',s))
    
    mr1_id = ['http://id.nlm.nih.gov/mesh/vocab#broaderConcept']
    
    mr2_id = ['http://www.w3.org/2000/01/rdf-schema#subClassOf']
    
    taxons = set([s for s,p,o in kg2 if p=='http://www.w3.org/1999/02/22-rdf-syntax-ns#type' and o=='https://www.ncbi.nlm.nih.gov/taxonomy/Taxon'])
    kg2 = [(s,p,o) for s,p,o in kg2 if s in taxons and o in taxons]
    
    models = ['DistMult','TransE']#,'HolE','ComplEx','ConvE','ConvKB','HAKE','RotatE','pRotate']
    
    embeddings_dir = 'pretrained_alt2/'
    results_dir = 'pretrained_results_alt2/'
    
    d = {}
    
    for model1, model2 in product(models,models):
        scores = []
        
        for kg,mr_id,kg_n,model in zip([kg1,kg2],[mr1_id,mr2_id],['chemical','taxonomy'],[model1,model2]):
            f = embeddings_dir+model+'_'+kg_n+'_embeddings.npy'
            embeddings = np.load(f)
            f = embeddings_dir+model+'_'+kg_n+'_ids.npy'
            ids = dict(np.load(f))
            edges = [(int(ids[s]),int(ids[o])) for s,p,o in kg if p in mr_id]
            nodes = create_nodes(edges,embeddings)
            leaf_nodes = [n for n in nodes if len(n.children) < 1]
            score = generate_score(leaf_nodes)/len(kg)
            scores.append(score)
        print(model1,model2,scores)
        df = pd.read_csv(results_dir+'_'.join([model1,model2,'1'])+'.csv',index_col='metric',usecols=['metric','value'])
        d[model1+model2]=sum([(1-float(df.loc['f1']))*s for s in scores])
    print(d)
    
def load_hier_embeddings(f,entities=None):
    df = pd.read_csv(f)
    cols = list(df.columns[1:-1])
    entities = entities or cols
    out = defaultdict(list)
    for c1 in cols:
        i = entities.index(c1)
        for c2 in cols:
            j = entities.index(c2)
            out[c1].append(df.iloc[cols.index(c1)+1,cols.index(c2)+1])
    return out
          
def test_hier_embeddings():
    for f,v in zip(['./embeddings/inorganic_embeddings.csv','./embeddings/Organic_chemicals_embeddings.csv','./embeddings/taxonomy_hierarchy_only0_embeddings.csv'],['./data/Inorganic_chemicals_hierarchy.csv','./data/Organic_chemicals_hierarchy.csv','./data/taxonomy_hierarchy_only0.csv']):
        embeddings = load_hier_embeddings(f)
        ids = dict([(a,b) for b,a in enumerate(embeddings)])
        embeddings = np.asarray([embeddings[k] for k in embeddings])
        
        kg = pd.read_csv(v)
    
        kg = list(zip(kg['subject'], kg['predicate'], kg['object']))
        g = nx.DiGraph()
        for s,p,o in kg:
            if p == 'http://www.w3.org/2000/01/rdf-schema#subClassOf':
                g.add_edge(s,o)
                
        edges = [(int(ids[i]),int(ids[j])) for i,j in g.edges if i in ids and j in ids]
        nodes = create_nodes(edges,embeddings)
        leaf_nodes = [n for n in nodes if len(n.children) < 1]
        score = generate_score(leaf_nodes)
        print(score)
          
if __name__ == '__main__':
    test_hier_embeddings()
    #test()
    #main()
    
    

    





