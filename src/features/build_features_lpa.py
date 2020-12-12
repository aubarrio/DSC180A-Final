from src.utils.utils_lpa import *

def build_feats(data, edges, cora):
    if cora:
        labels, idx, X = parse_data(data)
        features = build_features(X)
        labels = encode_label(labels)
        edges = build_edges(idx, parse_edges(edges))
        adj = build_adj(edges, labels)
        labels_for_lpa = torch.from_numpy(labels).type(torch.FloatTensor)
        labels = torch.LongTensor(np.where(labels)[1])
        idx_train, idx_val, idx_test = build_idx(X.shape[0])
        return features, labels, adj, labels_for_lpa, idx_train, idx_val
    else:
        raise NotImplementedError()
