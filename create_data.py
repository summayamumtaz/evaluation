## create data

"""
Steps:
1. Part
    1.1. Load ncbi taxonomy.
    1.2. Load ecotox effects.
    1.3. Use ncbi to ecotox mapping to replace species ids in ecotox effects.
    1.4. Use pubchem to cas mapping to replace chemical ids in ecotox effects.

2. Part
    2.1. Identify all species and chemicals used in relevant effects (eg. LC50).
    2.2. Find all triples in ncbi taxonomy which is connected to any effect species.
    2.3. Find all triples in chemble/pubchem/mesh which is connected to any effect chemical. 
    
3. Part
    3.1. Export effect data as tuples (chemical, species, concentration) .
    3.2. Export taxonomy triples as tuples (subject,predicate,object) .
    3.3. Export chemical triples as tuples (subject,predicate,object) .
    
"""

from tera.DataAggregation import Taxonomy, Effects, Traits
from tera.DataAccess import EffectsAPI
from tera.DataIntegration import DownloadedWikidata, LogMapMapping
from tera.utils import strip_namespace, unit_conversion
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal, BNode
import pandas as pd
import pubchempy as pcp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import networkx as nx
from rdflib.namespace import RDFS

import re

EPSILON = 10e-9

def get_subgraph(to_visit, graph, backtracking=0):
    out = set()
    visited = set()
    while to_visit:
        curr = to_visit.pop()
        visited.add(curr)
        tmp = set(graph.triples((curr,None,None)))
        out |= set([(s,p,o) for s,p,o in tmp if not isinstance(o,Literal)])
        to_visit |= set([o for _,_,o in tmp if not isinstance(o,Literal)])
        to_visit -= visited
        
    if backtracking > 0:
        tmp = set()
        for s in set([s for s,_,_ in out]):
            tmp |= set(graph.subjects(object=s))
        for t in out:
            graph.remove(t)
        return out | get_subgraph(tmp, graph, backtracking-1)
        
    return out


def get_longest_path(s,root,graph,p):
    #return triples in longest path to root:
    
    q = """select ?u ?v {
	<%s> <%s>* ?u .
	?u ?p ?v .
	?v <%s>* <%s> .

        } """ % (str(s),str(p),str(p),str(root))
    
    results = graph.query(q)
    g = nx.DiGraph()
    for u,v in results:
        g.add_edge(str(u),str(v))
    
    longest_path = []
    try:
        for path in nx.all_simple_paths(g, source=str(s), target=str(root)):
            if len(path) > len(longest_path):
                longest_path = path
    except nx.exception.NodeNotFound:
        pass
    
    return longest_path
    

def plot_data(filename):
    df = pd.read_csv(filename)
    y = np.asarray(df['concentration'])
    plt.hist(y,bins='auto')
    plt.show()

def main():
    #PART 1
    t = Taxonomy(directory='../taxdump/', verbose=False)
    
    ne = DownloadedWikidata(filename='./ncbi_to_eol.csv', verbose=False)
    
    n = list(set(t.graph.subjects(predicate=t.namespace['rank'],
                                object=t.namespace['rank/species'])))

    tr = Traits(directory='../eol/', verbose=False)
    conv = ne.convert(n, strip=True)
    converted = [(tr.namespace[i],k) for k,i in conv.items() if i != 'no mapping']

    tr.replace(converted)
    
    ed = Effects(directory='../ecotox_data/',verbose=False)
    
    n = list(set(t.graph.subjects(predicate=t.namespace['rank'], object=t.namespace['rank/species'])))
    
    species = LogMapMapping(filename='./data/consensus_mappings.txt',strip=True).convert(n,strip=True,reverse=True)
    species = [(ed.namespace['taxon/'+i],k) for k,i in species.items() if i != 'no mapping']
    
    ed.replace(species)
    
    n = list(set(ed.graph.objects(predicate=ed.namespace['chemical'])))
    chemicals = DownloadedWikidata(filename='./cas_to_mesh.csv').convert(n,reverse=False,strip=True)
    chemicals = [(k,URIRef('http://id.nlm.nih.gov/mesh/'+str(i))) for k,i in chemicals.items() if i != 'no mapping']
    
    ed.replace(chemicals)
    
    print('Part 1. done.')
    
    _,species = zip(*species)
    _,chemicals = zip(*chemicals)
    
    chemicals = set(map(str, chemicals))
    species = set(map(str, species))
    
    endpoints = EffectsAPI(dataobject=ed, verbose=True).get_endpoint(c=None, s=None)
             
    effects = [str(ed.namespace['effect/'+ef]) for ef in ['MOR','NER','DVP','GRO','IMM','INJ','ITX','MPH','PHY','REP']]
    d = defaultdict(list)
    for c,s,cc,cu,ep,ef,sd,sdu in tqdm(endpoints):
        if str(s) in species and str(c) in chemicals:
            if str(ef) in effects:
                try:
                    factor = unit_conversion(str(cu),'http://qudt.org/vocab/unit#MilligramPerLitre')
                except:
                    factor = 0
                    
                if factor > 0:
                    cc = float(cc)
                    cc = cc*factor
                    cc = np.log(cc+EPSILON)
                    
                    ep = str(ep).split('/')[-1]
                    try:
                        num = float('.'.join([re.findall(r'\d+', s).pop(0) for s in ep.split('.')]))
                    except IndexError:
                        continue
                    
                    d['degree'].append(num/100)
                    d['chemical'].append(str(c))
                    d['species'].append(str(s))
                    d['concentration'].append(cc)
                    d['effect'].append(str(ef))
    
    df = pd.DataFrame(data=d)
    df.to_csv('./data/data.csv')
    
    print('Part 2. done.')
    
    backsteps = 0
    
    for i in range(backsteps+1):
        tmp = set([URIRef(a) for a in set(df['species'])])
        triples = get_subgraph(tmp, t.graph+tr.graph, backtracking=i)
        s,p,o = zip(*triples)
        data = {'subject':s, 
                'predicate':p, 
                'object':o}
        df = pd.DataFrame(data=data)
        df.to_csv('./data/taxonomy'+str(i)+'.csv')
        
        
    root = 'https://www.ncbi.nlm.nih.gov/taxonomy/taxon/1'
    triples = []
    root = URIRef(root)
    for a in tqdm(set(df['species'])):
        a = URIRef(a)
        path = get_longest_path(a,root,t.graph,p=RDFS.subClassOf)
        for i in range(len(path)-1):
            triples.append((path[i],RDFS.subClassOf,path[i+1]))
    s,p,o = zip(*triples)
    data = {'subject':s, 
            'predicate':p, 
            'object':o}
    df = pd.DataFrame(data=data)
    df.to_csv('./data/taxonomy_hierarchy_only.csv')
        
    exit()
        
    df = pd.read_csv('./data/data.csv')
    mesh_graph = Graph()
    mesh_graph.parse('../pubchem/mesh.nt',format='nt')
    
    all_elements = set()
    
    for root, name in zip(['http://id.nlm.nih.gov/mesh/D007287',
                           'http://id.nlm.nih.gov/mesh/D009930',
                           'http://id.nlm.nih.gov/mesh/D006571',
                           'http://id.nlm.nih.gov/mesh/D011083',
                           'http://id.nlm.nih.gov/mesh/D046911',
                           'http://id.nlm.nih.gov/mesh/D006730',
                           'http://id.nlm.nih.gov/mesh/D045762',
                           'http://id.nlm.nih.gov/mesh/D002241',
                           'http://id.nlm.nih.gov/mesh/D008055',
                           'http://id.nlm.nih.gov/mesh/D000602',
                           'http://id.nlm.nih.gov/mesh/D009706',
                           'http://id.nlm.nih.gov/mesh/D045424',
                           'http://id.nlm.nih.gov/mesh/D001685',
                           'http://id.nlm.nih.gov/mesh/D001697',
                           'http://id.nlm.nih.gov/mesh/D004364',
                           'http://id.nlm.nih.gov/mesh/D020164'],
                            ['Inorganic_chemicals',
                              'Organic_chemicals',
                              'Hetrocyclic_compounds',
                              'Polycyclic_compounds',
                              'Macromolecular_substances',
                              'Hormones_Hormone_Substitutes_Hormone_Antagonists',
                              'Enzymes_and_Coenzymes',
                              'Carbohydrates',
                              'Lipids',
                              'Amino_Acids_Peptides_Proteins',
                              'Nucleic_Acids_Nucleotides_Nucleosides',
                              'Complex_Mixtures',
                              'Biological_Factors',
                              'Biomedical_Dental_Materials',
                              'Pharmaceutical_Preparations',
                              'Chemical_Actions_Uses']):
        triples = []
        root = URIRef(root)
        for a in tqdm(set(df['chemical'])):
            a = URIRef(a)
            path = get_longest_path(a,root,mesh_graph,p='http://id.nlm.nih.gov/mesh/vocab#broaderDescriptor')
            for i in range(len(path)-1):
                triples.append((path[i],RDFS.subClassOf,path[i+1]))
        if len(triples) > 0:
            s,p,o = zip(*triples)
            data = {'subject':s, 
                    'predicate':p, 
                    'object':o}
            df = pd.DataFrame(data=data)
            df.to_csv('./data/'+name+'_hierarchy.csv')
        
            all_elements |= set(s)
            all_elements |= set(o)
    
    for i in range(backsteps+1):
        triples = get_subgraph(set([URIRef(a) for a in all_elements]), mesh_graph, backtracking=i)
        s,p,o = zip(*triples)
        data = {'subject':s, 
                'predicate':p, 
                'object':o}
        df = pd.DataFrame(data=data)
        df.to_csv('./data/chemicals'+str(i)+'.csv')
    
    print('Part 3. done.')
    
    
if __name__ == '__main__':
    main()
