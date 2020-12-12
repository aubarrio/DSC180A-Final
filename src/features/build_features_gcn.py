from src.utils.utils_gcn import *

def build_feats(data_graph, edges_label, cora):
    if cora:
        data = data_graph
        edges = edges_label
        #this means we have data and edges
        labels, nodes, X, N, F = parse_data(data)
        edge_list = parse_edges(edges)

        train_idx, val_idx, test_idx = limit_data(labels)

        train_mask = np.zeros((N,),dtype=bool)
        train_mask[train_idx] = True

        val_mask = np.zeros((N,),dtype=bool)
        val_mask[val_idx] = True

        test_mask = np.zeros((N,),dtype=bool)
        test_mask[test_idx] = True

        labels_encoded, classes = encode_label(labels)

        A = build_adj(nodes, edge_list)

        return A, F, N, X, train_mask, val_mask, labels_encoded, classes
    else:
        graph = data_graph
        label = edges_label

        labels = [i[0] for i in label]

        nodes = list(range(graph['num_nodes']))

        X = graph['node_feat']

        N = X.shape[0]
        F = X.shape[1]

        edge_list = np.array(list(map(list, zip(graph['edge_index'][0],
                        graph['edge_index'][1]))), dtype=np.int32)

        train_idx,val_idx,test_idx = build_idx(N)

        train_mask = np.zeros((N,),dtype=bool)
        train_mask[train_idx] = True

        val_mask = np.zeros((N,),dtype=bool)
        val_mask[val_idx] = True

        test_mask = np.zeros((N,),dtype=bool)
        test_mask[test_idx] = True

        labels_encoded, classes = encode_label(labels)

        A = build_adj(nodes, edge_list)

        return A, F, N, X, train_mask, val_mask, labels_encoded, classes
