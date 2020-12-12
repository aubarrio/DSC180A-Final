import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import torch
import pandas as pd

def parse_data(data):
    labels, nodes, X = [], [], []
    for i, data in enumerate(data):
        features = data.split('\t')
        labels.append(features[-1])
        X.append(features[1:-1])
        nodes.append(features[0])

    X = np.array(X, dtype=float)
    # X = np.array(X, dtype=int)
    return labels, np.array(nodes, dtype=np.int32), X

def parse_edges(edges):
    edge_list = []
    for edge in edges:
        e = edge.split('\t')
        edge_list.append([e[0],e[1]])
    return np.array(edge_list).astype('int32')

def build_features(X):
    features = sp.csr_matrix(X, dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    return features

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels.astype('int32')

def build_edges(idx, edges):
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges.flatten())),
                        dtype=np.int32).reshape(edges.shape)
    return edges

def build_adj(edges, labels):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
              shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def build_idx(shape):
    x1 = int(0.6 * shape)
    x2 = x1 + int(0.2 * shape)
    x3 = x2 + int(0.2 * shape)
    idx_train = torch.LongTensor(range(x1))
    idx_val = torch.LongTensor(range(x1, x2))
    idx_test = torch.LongTensor(range(x2, x3 - 1))
    return idx_train, idx_val, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
