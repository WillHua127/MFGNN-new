import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
import os
import re
import torch as th
from sklearn.model_selection import ShuffleSplit
from numpy.linalg import matrix_power
import dgl
import scipy.io
import csv
from os import path
from sklearn.preprocessing import label_binarize
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.model_selection import StratifiedKFold
from dgl import backend
from ogb.nodeproppred import PygNodePropPredDataset

sys.setrecursionlimit(99999)

    
def load_fixed_splits(dataset, sub_dataset):
    name = dataset
    if sub_dataset:
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], \
            dest_path=f'./data/splits/{name}-splits.npy', showsize=True) 
    
    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

def random_disassortative_splits(labels, num_classes):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels, num_classes
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(torch.round(0.6*torch.div(labels.size()[0], num_classes)))
    val_lb = int(round(0.2*labels.size()[0]))
    # train_index = torch.cat([i[:int(len(i)*0.6)] for i in indices], dim=0)
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    # val_index = torch.cat([i[int(len(i)*0.6):int(len(i)*0.8)] for i in indices], dim=0)
    # test_index = torch.cat([i[int(len(i)*0.8):] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    val_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]
    
    
    return train_index, val_index, test_index



def semi_supervised_splits(dataset):
    if dataset =='cora':
        train_index, val_index, test_index = range(140), range(140, 640), range(1708, 2708)
    elif dataset =='citeseer':
        train_index, val_index, test_index = range(120), range(120, 620), range(2312, 3312)
    elif dataset =='pubmed':
        train_index, val_index, test_index = range(60), range(60, 560), range(18717, 19717)
        
    train_index = torch.LongTensor(train_index)
    val_index = torch.LongTensor(val_index)
    test_index = torch.LongTensor(test_index)
        
    #idx_train = range(y)
    #idx_val = range(y, y+500)
    #idx_test = range(y+500, y+1500)
    
    return train_index, val_index, test_index

    
def data_split(idx, dataset_name):
    splits_file_path = 'splits/'+dataset_name+'_split_0.6_0.2_'+str(idx)+'.npz'
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)
    return train_mask, val_mask, test_mask

def normalize_sp(spmx):
    rowsum = sp.csr_matrix(spmx.sum(axis=1))
    r_inv= sp.csr_matrix.power(rowsum, -1)
    #r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.transpose()
    scaling_matrix = sp.diags(r_inv.toarray()[0])
    spmx = scaling_matrix.dot(spmx)
    return spmx

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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def load_graph_data(dataset_name):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
    elif dataset_name in {'deezer'}:
        adj, features, labels = load_deezer_dataset()
        
    elif dataset_name in {'yelpchi'}:
        adj, features, labels = load_yelpchi_dataset()
        
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                f'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        #print(type(adj))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    
    adj.setdiag(1)
    g = dgl.DGLGraph(adj)
    
    features = preprocess_features(features)
    #features = np.hstack([features, np.ones([features.shape[0],1])])
    
    
    num_labels = len(np.unique(labels))
    #onehot_labels = np.eye(num_labels)[labels]
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    #print(features.shape)
    if dataset_name in {'deezer', 'yelpchi'}:
        #eatures = normalize_sp(features)
        features = sparse_mx_to_torch_sparse_tensor(features).to_dense()
    else:
        #features = preprocess_features(features)
        features = th.FloatTensor(features)
        
    #features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    g.ndata['features'] = features
    g.ndata['labels'] = labels
    
    #degs = g.in_degrees().float()
    #norm = torch.pow(degs, -0.5)
    #norm[torch.isinf(norm)] = 0
    #g.ndata['norm'] = norm.unsqueeze(1)
    
    return g, num_labels 

def separate_data(graph_list, labels, seed, fold_idx):
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    train_label_list = [labels[i] for i in train_idx]
    test_label_list = [labels[i] for i in test_idx]
    train_label_list = torch.LongTensor(train_label_list)
    test_label_list = torch.LongTensor(test_label_list)

    return train_graph_list, test_graph_list, train_label_list, test_label_list


def load_gc_data(dataset):
    print('loading data')
    graphs = []
    glabel_dict = {}
    nlabel_dict = {}
    elabel_dict = {}
    ndegree_dict = {}
    labels = []
    nattrs_flag = False
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())

        for i in range(n_g):
            grow = f.readline().strip().split()
            n_nodes, glabel = [int(w) for w in grow]

            if not glabel in glabel_dict:
                mapped = len(glabel_dict)
                glabel_dict[glabel] = mapped

            labels.append(glabel_dict[glabel])

            #g = dgl.DGLGraph(([], []))
            g = dgl.DGLGraph()
            g.add_nodes(n_nodes)

            nlabels = []  # node labels
            nattrs = []  # node attributes if it has
            m_edges = 0

            for j in range(n_nodes):
                nrow = f.readline().strip().split()

                # handle edges and attributes(if has)
                tmp = int(nrow[1]) + 2  # tmp == 2 + #edges
                if tmp == len(nrow):
                    # no node attributes
                    nrow = [int(w) for w in nrow]
                elif tmp > len(nrow):
                    nrow = [int(w) for w in nrow[:tmp]]
                    nattr = [float(w) for w in nrow[tmp:]]
                    nattrs.append(nattr)
                else:
                    raise Exception('edge number is incorrect!')

                if not nrow[0] in nlabel_dict:
                    mapped = len(nlabel_dict)
                    nlabel_dict[nrow[0]] = mapped

                nlabels.append(nlabel_dict[nrow[0]])

                m_edges += nrow[1]
                g.add_edges(j, nrow[2:])


            if nattrs != []:
                nattrs = np.stack(nattrs)
                g.ndata['attr'] = backend.tensor(nattrs, backend.float32)
                nattrs_flag = True

            g.ndata['label'] = backend.tensor(nlabels)

            assert g.number_of_nodes() == n_nodes


            graphs.append(g)

    #labels = F.tensor(labels)
    if not nattrs_flag:
        nlabel_set = set([])
        for g in graphs:
            nlabel_set = nlabel_set.union(set([backend.as_scalar(nl) for nl in g.ndata['label']]))
        nlabel_set = list(nlabel_set)
        is_label_valid = all([label in nlabel_dict for label in nlabel_set])
        if is_label_valid and len(nlabel_set) == np.max(nlabel_set) + 1 and np.min(nlabel_set) == 0:
            label2idx = nlabel_dict
        else:
            label2idx = {
                nlabel_set[i]: i
                for i in range(len(nlabel_set))
            }
        for g in graphs:
            attr = np.zeros((g.number_of_nodes(), len(label2idx)))
            attr[range(g.number_of_nodes()), [label2idx[nl]
                                              for nl in backend.asnumpy(g.ndata['label']).tolist()]] = 1
            g.ndata['attr'] = backend.tensor(attr, backend.float32)
    return graphs, labels, len(nlabel_dict)


def load_ogb_graph(dataset_name):
    if not os.path.isfile('torch_geometric_data/dgl_'+dataset_name):
        dataset = PygNodePropPredDataset(name = "ogbn-"+dataset_name, root = 'torch_geometric_data/')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        edge = dataset[0].edge_index
        num_classes = len(np.unique(dataset[0].y))
        print("Nodes: %d, edges: %d, features: %d, classes: %d. \n"%(dataset[0].y.shape[0], len(edge[0])/2, len(dataset[0].x[0]), num_classes))
        graph = dgl.DGLGraph((edge[0],edge[1]))
        graph.ndata['features'] = dataset[0].x
        graph.ndata['labels'] = dataset[0].y
        dgl.data.utils.save_graphs('torch_geometric_data/dgl_'+dataset_name, graph)
        torch.save(train_idx, 'torch_geometric_data/ogbn_'+dataset_name+'/train_'+dataset_name+'.pt')
        torch.save(valid_idx, 'torch_geometric_data/ogbn_'+dataset_name+'/valid_'+dataset_name+'.pt')
        torch.save(test_idx, 'torch_geometric_data/ogbn_'+dataset_name+'/test_'+dataset_name+'.pt')
        labels = graph.ndata.pop('labels')
        features = graph.ndata.pop('features')
    elif os.path.isfile('torch_geometric_data/dgl_'+dataset_name):
        graph = dgl.data.utils.load_graphs('torch_geometric_data/dgl_'+dataset_name)[0][0]
        labels = graph.ndata.pop('labels')
        features = graph.ndata.pop('features')
        train_idx = torch.load('torch_geometric_data/ogbn_'+dataset_name+'/train_'+dataset_name+'.pt')
        valid_idx = torch.load('torch_geometric_data/ogbn_'+dataset_name+'/valid_'+dataset_name+'.pt')
        test_idx = torch.load('torch_geometric_data/ogbn_'+dataset_name+'/test_'+dataset_name+'.pt')
        num_classes = len(torch.unique(labels))
        
    return graph, features, labels, num_classes, train_idx, valid_idx, test_idx


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels

def load_torch_geometric_data(dataset, name):
    cur = os.getcwd()

    if dataset in {'WikiCS', 'Flickr'}:
        data = eval(dataset + "(root = '" + cur.replace("\\", "/") + "/torch_geometric_data/" + dataset + "')")
    else:
        data = eval(dataset + "(root = '" + cur.replace("\\", "/") + "/torch_geometric_data/" + dataset + "'," + "name = '" + name + "')")
    # e.g. Coauthor(root='...', name = 'CS')

    edge = data[0].edge_index
    if contains_self_loops(edge):
        edge = remove_self_loops(edge)[0]
        print("Original data contains self-loop, it is now removed")

    adj = to_dense_adj(edge)[0].numpy()

    print("Nodes: %d, edges: %d, features: %d, classes: %d. \n"%(len(adj[0]), len(edge[0])/2, len(data[0].x[0]), len(np.unique(data[0].y))))

    mask = np.transpose(adj) != adj
    col_sum = adj.sum(axis=0)
    print("Check adjacency matrix is sysmetric: %r"%(mask.sum().item() == 0))
    print("Chenck the number of isolated nodes: %d"%((col_sum == 0).sum().item()))
    print("Node degree Max: %d, Mean: %.4f, SD: %.4f"%(col_sum.max(), col_sum.mean(), col_sum.std()))

    return adj, data[0].x.numpy(), data[0].y.numpy()


def load_deezer_dataset():
    deezer = scipy.io.loadmat(f'data/deezer-europe.mat')
    A, label, features = deezer['A'], deezer['label'], deezer['features']

    return A, features, label[0]


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"data/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = sp.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0]#.reshape(n,2514) # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    
    return A, features, label


def load_fb_dataset(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    if filename not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
        print('Invalid sub_dataname, deferring to Penn94 graph')
        sub_dataname = 'Penn94'
    mat = scipy.io.loadmat('./data/facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info'].astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))
    print(features.shape)
    print(np.where(label==-1))
        
    return A, features, label


def load_yelpchi_dataset():
    if not path.exists(f'./data/yelpchi.mat'):
            gdd.download_file_from_google_drive(
                file_id= dataset_drive_url['yelp-chi'], \
                dest_path=f'./data/yelpchi.mat', showsize=True) 
    fulldata = scipy.io.loadmat(f'./data/yelpchi.mat')
    A = fulldata['homo']
    features = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()

    return A, features, label


def load_pokec_mat():
    """ requires pokec.mat """
    if not path.exists(f'./data/pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id= dataset_drive_url['pokec'], \
            dest_path=f'./data/pokec.mat', showsize=True) 

    fulldata = scipy.io.loadmat(f'./data/pokec.mat')

    fulldata = scipy.io.loadmat(f'./data/snap_patents.mat')
    edge_index = fulldata['edge_index']
    features = fulldata['node_feat']
    n = features.shape[0]
    (src, tar) = edge_index
    A = sp.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(tar))),
                                shape=(n,n))
    label = fulldata['label'].flatten()

    return A, features, label

def load_snap_mat(nclass=5):
    if not path.exists(f'./data/snap_patents.mat'):
        gdd.download_file_from_google_drive(
            file_id= dataset_drive_url['snap-patents'], \
            dest_path=f'./data/snap_patents.mat', showsize=True) 

    fulldata = scipy.io.loadmat(f'./data/snap_patents.mat')
    edge_index = fulldata['edge_index']
    features = fulldata['node_feat']
    n = features.shape[0]
    (src, tar) = edge_index
    A = sp.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(tar))),
                                shape=(n,n))
    
    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    
    return A, features, label


def full_load_data(dataset_name, sub_dataname=''):
    #splits_file_path = 'splits/'+dataset_name+'_split_0.6_0.2_'+str(idx)+'.npz'
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)

    elif dataset_name in {'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Computers', 'Amazon_Photo'}:
        dataset, name = dataset_name.split("_")
        adj, features, labels = load_torch_geometric_data(dataset, name)

    elif dataset_name in {'Flickr', 'WikiCS'}:
        adj, features, labels = load_torch_geometric_data(dataset_name, None)
    
    elif dataset_name in {'twitch'}:
        adj, features, labels = load_twitch_dataset(sub_dataname)
        
    elif dataset_name in {'facebook'}:
        adj, features, labels = load_fb_dataset(sub_dataname)
        
    elif dataset_name in {'deezer'}:
        adj, features, labels = load_deezer_dataset()
        
    elif dataset_name in {'yelpchi'}:
        adj, features, labels = load_yelpchi_dataset()
    
    elif dataset_name in {'snap'}:
        adj, features, labels = load_snap_mat()
        
    elif dataset_name in {'pokec'}:
        adj, features, labels = load_pokec_mat()
        
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        #G = nx.DiGraph()
        G = nx.Graph()
        #G = dgl.DGLGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        
        
        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])
                    
                    
                    
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))
                
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        #print(len(G.edges))
     

    adj.setdiag(1)
    edge_dict = {}
    for i in range(adj.shape[0]):
        edge_dict[i]=sp.find(adj[i])[1]
    #print(np.arange(len(np.unique(labels))))
    
    
    if dataset_name in {'deezer', 'yelpchi', 'snap', 'pokec'}:
        features = normalize_sp(features)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = preprocess_features(features)
        features = np.hstack([features, np.ones([features.shape[0],1])])
        #features = pre_aggregate(features, edge_dict)
        features = th.FloatTensor(features)
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    labels = th.LongTensor(labels)
    # train_mask = th.BoolTensor(train_mask)
    # val_mask = th.BoolTensor(val_mask)
    # test_mask = th.BoolTensor(test_mask)

    adj = normalize_adj(adj+sp.eye(adj.shape[0]))
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    #print(isinstance(edge_index, torch.Tensor))
    
    return edge_dict, features, labels, edge_index#, train_mask, val_mask, test_mask




dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
}

    
    
def pre_aggregate(feats, edge_dict):
    trans_feats = feats.copy()
    for i in range(feats.shape[0]):
            trans_feats[i] = (np.prod(feats[edge_dict[i]],axis=0))
    return trans_feats
