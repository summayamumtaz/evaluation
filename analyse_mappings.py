
from rdflib import Graph, Literal
from rdflib.namespace import OWL, RDFS
from tera.DataIntegration import LogMapMapping, Alignment

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm

EPSILON = np.finfo(np.float32).eps

def precision(refrence, alignment):
    return len(refrence.intersection(alignment))/(len(alignment) + EPSILON)

def recall(refrence, alignment):
    return len(refrence.intersection(alignment))/(len(refrence) + EPSILON)

def f_measure(refrence, alignment, b = 1):
    p = precision(refrence, alignment)
    r = recall(refrence, alignment)
    return (1+b**2)*(p*r)/(b**2*p + r)

def reverse_mapping(mapping):
    out = {}
    for k in mapping:
        if isinstance(mapping[k],(list,set)):
            for a in mapping[k]:
                out[a] = [k]
        else:
            out[mapping[k]] = [k]
    return out

def make_mapping_unique(mapping):
    return {k:mapping[k].pop() for k in mapping}

def load_mapping(f, th=0.8, filtered=False, ref1=[], ref2=[], unique=True):
    final_mapping1 = set()
    final_mapping2 = set()
    m = LogMapMapping(f, strip=False, threshold=th, unique=unique)
    m.load()
    um = m.mappings
    if unique:
        um = make_mapping_unique(um)
    for k in um:
        if filtered:
            if not k in ref1: continue
        e = um[k]
        if unique and not isinstance(e,list):
            e = [e]
        for a in e:
            final_mapping1.add((k,a,m.scores[(k,a)]))
    
    rm = reverse_mapping(um)
    if unique:
        rm = make_mapping_unique(rm)
    for k in rm:
        if filtered:
            if not k in ref2: continue
        e = rm[k]
        if unique and not isinstance(e,list):
            e = [e]
        for a in e:
            final_mapping2.add((k,a,m.scores[(a,k)]))
            
    return final_mapping1, final_mapping2

def make_unique(fm, k = 1):
    tmp = defaultdict(list)
    for e1,e2,score in fm:
        tmp[e1].append((e2,score))
    
    out = set()
    for e1 in tmp:
        sorted_tmp = sorted(tmp[e1], key=lambda x: x[1], reverse=True)
        for _ in range(k):
            e2,_ = sorted_tmp.pop(0)
            out.add((e1,e2))
    return out

def main(filtered=True,unique=True,top_k=1):

    g = Graph()
    g.parse('../rdf/partial_aligment_ecotox_ncbi.nt',format='nt')
    reference_mapping1 = set()
    reference_mapping2 = set()
    
    unique = unique and top_k < 2
    
    ref1 = set()
    ref2 = set()
    
    for s,o in g.subject_objects(predicate=OWL.sameAs):
        reference_mapping1.add((str(s),str(o)))
        reference_mapping2.add((str(o),str(s)))
        ref1.add(str(s))
        ref2.add(str(o))
    
    print('Number of refrerence mappings: ',len(reference_mapping1))
    
    metrics = {'logmap_outputs/':{'precision':[],'recall':[],'f1':[]},
               'aml_output/':{'precision':[],'recall':[],'f1':[]},
               'string_matcher_output/':{'precision':[],'recall':[],'f1':[]},
               }
    
    r = np.arange(0.5,1,0.1)
    for th in tqdm(r):
        for d,fn in zip(['logmap_outputs/','aml_output/','string_matcher_output/'],['/logmap_mappings.txt','.rdf','.txt']):
            final_mapping1 = set()
            final_mapping2 = set()
            for part in range(0,12):
                f = filename = d+str(part)+fn
                tmp1, tmp2 = load_mapping(f,th=th,filtered=filtered,ref1=ref1,ref2=ref2,unique=unique)
                final_mapping1 |= tmp1
                final_mapping2 |= tmp2
            
            if unique:
                final_mapping1 = make_unique(final_mapping1,k=top_k)
                final_mapping2 = make_unique(final_mapping2,k=top_k)
            else:
                final_mapping1 = set([(a,b) for a,b,c in final_mapping1])
                final_mapping2 = set([(a,b) for a,b,c in final_mapping2])
            
            metrics[d]['precision'].append((precision(reference_mapping1,final_mapping1)+precision(reference_mapping2,final_mapping2))/2)
            metrics[d]['recall'].append((recall(reference_mapping1,final_mapping1)+recall(reference_mapping2,final_mapping2))/2)
            metrics[d]['f1'].append((f_measure(reference_mapping1,final_mapping1)+f_measure(reference_mapping2,final_mapping2))/2)
            
    best_th = {}
    for k in metrics:
        best_th[k] = r[np.argmax(metrics[k]['f1'])]
    
    final_mapping1 = defaultdict(set)
    final_mapping2 = defaultdict(set)
    for d,fn in zip(best_th,['/logmap_mappings.txt','.rdf','.txt']):
        for part in range(0,12):
            f = filename = d+str(part)+fn
            tmp1, tmp2 = load_mapping(f,th=best_th[d],filtered=True,ref1=ref1,ref2=ref2,unique=unique)
            final_mapping1[d] |= tmp1
            final_mapping2[d] |= tmp2
    
    if unique:
        for k in final_mapping1:
            final_mapping1[k] = make_unique(final_mapping1[k],k=top_k)
        for k in final_mapping2:
            final_mapping2[k] = make_unique(final_mapping2[k],k=top_k)
    else:
        for k in final_mapping1:
            final_mapping1[k] = set([(a,b) for a,b,c in final_mapping1[k]])
        for k in final_mapping2:
            final_mapping2[k] = set([(a,b) for a,b,c in final_mapping2[k]])
            
    final_mapping1['agreement'] = set.intersection(*[final_mapping1[k] for k in final_mapping1])
    final_mapping2['agreement'] = set.intersection(*[final_mapping2[k] for k in final_mapping2])
    final_mapping1['union'] = set.union(*[final_mapping1[k] for k in final_mapping1])
    final_mapping2['union'] = set.union(*[final_mapping2[k] for k in final_mapping2])
    
    
    for d in final_mapping1:
        print(d,'precision',(precision(reference_mapping1,final_mapping1[d])+precision(reference_mapping2,final_mapping2[d]))/2)
        print(d,'recall',(recall(reference_mapping1,final_mapping1[d])+recall(reference_mapping2,final_mapping2[d]))/2)
        print(d,'f1',(f_measure(reference_mapping1,final_mapping1[d])+f_measure(reference_mapping2,final_mapping2[d]))/2)
    
    final_mapping1 = defaultdict(set)
    final_mapping2 = defaultdict(set)
    for d,fn in zip(best_th,['/logmap_mappings.txt','.rdf','.txt']):
        for part in range(0,12):
            f = d+str(part)+fn
            tmp1, tmp2 = load_mapping(f,th=best_th[d],filtered=False,unique=unique)
            final_mapping1[d] |= tmp1
            final_mapping2[d] |= tmp2
    
    if unique:
        for k in final_mapping1:
            final_mapping1[k] = make_unique(final_mapping1[k],k=top_k)
        for k in final_mapping2:
            final_mapping2[k] = make_unique(final_mapping2[k],k=top_k)
    else:
        for k in final_mapping1:
            final_mapping1[k] = set([(a,b) for a,b,c in final_mapping1[k]])
        for k in final_mapping2:
            final_mapping2[k] = set([(a,b) for a,b,c in final_mapping2[k]])
            
    final_mapping1['agreement'] = set.intersection(*[final_mapping1[k] for k in final_mapping1])
    final_mapping2['agreement'] = set.intersection(*[final_mapping2[k] for k in final_mapping2])
    final_mapping1['output'] = set.intersection(*[final_mapping1[k] for k in ['logmap_outputs/','aml_output/']])
    final_mapping2['output'] = set.intersection(*[final_mapping2[k] for k in ['logmap_outputs/','aml_output/']])
    final_mapping1['union'] = set.union(*[final_mapping1[k] for k in final_mapping1])
    final_mapping2['union'] = set.union(*[final_mapping2[k] for k in final_mapping2])
    
    for d1 in final_mapping1:
        print('# mapping',len(final_mapping1[d1]))
        for d2 in final_mapping1:
            tmp = final_mapping1[d1] - final_mapping1[d2]
            print('Disagreement ',d1,d2,len(tmp)/len(final_mapping1[d1]))
            
    
    print('Outputting consensus mappings.')
    
    with open('consensus_mappings.txt','w') as f:
        for e1,e2 in final_mapping1['output']:
            s = '|'.join([e1,e2,str(1)]) +'\n'
            f.write(s)

if __name__ == '__main__':
    main(filtered=True,unique=True,top_k=1)
    #main(filtered=True,unique=False)

