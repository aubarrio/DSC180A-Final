import numpy as np
import networkx as nx
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def parse_data(data):
    labels, nodes, X = [], [], []
    for i, data in enumerate(data):
        features = data.split('\t')

        labels.append(features[-1])
        X.append(features[1:-1])
        nodes.append(features[0])

    X = np.array(X, dtype=float)
    X = np.array(X, dtype=int)
    return labels, nodes, X, X.shape[0], X.shape[1]

def parse_edges(edges):
    edge_list = []
    for edge in edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))
    return edge_list

def limit_data(labels,limit=20,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1

        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break

    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    #get the first val_num
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx, test_idx

def build_idx(shape):
    x1 = int(0.6 * shape)
    x2 = x1 + int(0.2 * shape)
    x3 = x2 + int(0.2 * shape)
    idx_train = list(range(x1))
    idx_val = list(range(x1, x2))
    idx_test = list(range(x2, x3 - 1))
    return idx_train, idx_val, idx_test

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, len(label_encoder.classes_)

def build_adj(nodes, edge_list):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    #obtain the adjacency matrix (A)
    A = nx.adjacency_matrix(G)
    return A
