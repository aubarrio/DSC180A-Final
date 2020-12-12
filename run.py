import sys
import argparse
import json

if __name__ == '__main__':
    targets = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora', choices={'cora', 'ogb'},
                    help='Choice of which dataset your model of choice will be run on, from cora or ogb dataset')
    parser.add_argument('--model', type=str, default='gcn', choices={'gcn', 'graphsage', 'gcn_lpa'},
                    help='Choice of model [gcn, graphsage, gcn_lpa]')
    parser.add_argument('--aggr', type=str, default='mean', choices={'mean', 'max'},
                    help='Choice of aggregator will only be applied to GraphSage')

    args = parser.parse_args()

    from src.data.make_dataset import *

    if args.data == 'cora':
        fp = 'data/cora'
    else:
        fp = 'data/ogb/obgn-arxiv'

    data, edges, cora = get_data(fp)

    if args.model in ['gcn', 'graphsage']:
        from src.features.build_features_gcn import *
        A, F, N, X, train_mask, val_mask, labels_encoded, num_classes = build_feats(data, edges, cora)
        #get our config parameters
        with open('config/gcn_params.json') as fh:
            data_cfg = json.load(fh)
        #pass in our features for our model
        if args.model == 'gcn':
            from src.models.gcn import *
            GCN(A, F, N, X, train_mask, val_mask, labels_encoded, num_classes, args.aggr, **data_cfg)
        else:
            from src.models.graphsage import *
            GraphSage(A, F, N, X, train_mask, val_mask, labels_encoded, num_classes, args.aggr, **data_cfg)
    else:
        from src.features.build_features_lpa import *
        features, labels, adj, labels_for_lpa, idx_train, idx_val = build_feats(data, edges, cora)
        #get our config parameters
        with open('config/lpa_params.json') as fh:
            data_cfg = json.load(fh)
        #pass in our features for our model
        from src.models.gcn_lpa import *
        GCN_LPA(features, labels, adj, labels_for_lpa, idx_train, idx_val, **data_cfg)
