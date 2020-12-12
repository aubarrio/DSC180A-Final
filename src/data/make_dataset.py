"""
This file is tasked with making the necessary data for a GCN or GraphSage model
"""
import os
import json
from sklearn.utils import shuffle
from ogb.nodeproppred import NodePropPredDataset

# 'data/ogb/ogbn-arxiv' ogb file path

def get_data(fp):
    if 'cora' in fp:
        data, edges = [], []
        for ro, di, files in os.walk(fp):
            for file in files:
                if '.content' in file:
                    with open(os.path.join(ro, file),'r') as f:
                        data.extend(f.read().splitlines())
                elif 'cites' in file:
                    with open(os.path.join(ro, file),'r') as f:
                        edges.extend(f.read().splitlines())
        data = shuffle(data)
        return data, edges, True
    elif 'ogb' in fp:
        #this means we are dealing with the ogb dataset
        d = NodePropPredDataset('ogbn-arxiv', root=fp)

        graph, label = d[0]
        return graph, label, False
    else:
        raise NotImplementedError()
