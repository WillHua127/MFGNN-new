from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import scipy.stats as test
import torch.optim as optim
import matplotlib

from utils import load_data, accuracy, full_load_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default = 'cornell')
args = parser.parse_args()


def Svalue(adj, X, average_mode = 0, hop = 1): #average_mode = 0 no average; 1 symmetric weighted average; 2 scalar(sparsity) average.
    nnodes = adj.shape[0]
    adj_dense = adj.to_dense()
    adj_exp = adj_dense
    for i in range(hop-1):
        adj_exp = torch.mm(adj_exp,adj)
        
    adj_exp[adj_exp!=0] = 1
    adj_exp = adj_exp - torch.diag(torch.diag(adj_exp))
    D = torch.diag(torch.sum(adj_exp,1))
    #add self-loop to adj for isolated nodes
    #self_loop = torch.zeros(nnodes)
    #self_loop[torch.diag(D)==0]=1
    #adj_dense = adj_dense + torch.diag(self_loop)
    #D = D + torch.diag(self_loop)
    
    L=D-adj_exp
    one = torch.ones(nnodes,nnodes)
    total_degree = torch.sum(D.diag())
    if average_mode == 1:
        r_inv_G = torch.diag(D.diag().pow(-1/2))
        r_inv_G[torch.isinf(r_inv_G)]=0
        L_sym_G = torch.mm(torch.mm(r_inv_G, L), r_inv_G)
        
        r_inv_GC = torch.diag(((nnodes-1)*torch.ones(nnodes)-D.diag()).pow(-1/2))
        r_inv_GC[torch.isinf(r_inv_GC)]=0
        L_sym_GC = torch.mm(torch.mm(r_inv_GC, nnodes*torch.eye(nnodes)-one-L),r_inv_GC)
        
        E_G = torch.trace(torch.mm(torch.mm(X.transpose(0,1),L_sym_G), X))
        E_GC = torch.trace(torch.mm(torch.mm(X.transpose(0,1),L_sym_GC), X))
        S=E_G/(E_G+E_GC)
        
    else:
        E_G = torch.trace(torch.mm(torch.mm(X.transpose(0,1),L), X))
        E_GC = torch.trace(torch.mm(torch.mm(X.transpose(0,1),nnodes*torch.eye(nnodes)-one), X)) - E_G
        if average_mode == 0:
            
            S = E_G/(E_G+E_GC)
        elif average_mode == 2:
           
            S = (E_G/total_degree)/((E_G/total_degree) + E_GC/(nnodes*(nnodes-1)-total_degree))
       
    #print(S)
    return S, D

def dataset_stats(dataset, hop = 1, aggregated = 0): #aggregated is the number of aggregation operated on X
    
    adj, adj_high, features, labels = full_load_data(dataset)
    labels = torch.nn.functional.one_hot(labels).float()

    for i in range(aggregated):
        features = torch.mm(adj,features)
        labels = torch.mm(adj,labels)
        
    
    
    S_n_features, D = Svalue(adj,features,0,hop)
    S_n_labels, _ = Svalue(adj,labels,0,hop)
    
    S_sym_features, _ = Svalue(adj,features,1,hop)
    S_sym_labels, _ = Svalue(adj,labels,1,hop)
    
    S_s_features, _ = Svalue(adj,features,2,hop)
    S_s_labels, _ = Svalue(adj,labels,2,hop)
    
    print("Stats of %s hop %d aggregated %d, No Average S-value: Feature %.4f, Labels %.4f; Symmetric Average S-value: Feature %.4f, Labels %.4f; Scalar Average S-value: Feature %.4f, Labels %.4f; "%
          (dataset,hop,aggregated,S_n_features,S_n_labels,S_sym_features,S_sym_labels,S_s_features,S_s_labels))
    
    print("Degree Mean: %.4f, Degree Std: %.4f, Max Degree: %d, Number of Isolated Nodes: %d"%(torch.mean(D.diag()), torch.std(D.diag()), torch.max(D.diag()), torch.sum(D.diag()==0)))

def hypothesis_test(dataset, hop = 1):
    #test.ks_2samp, test.ttest_ind, test.chisquare, test.ranksums, test.wilcoxon, test.kruskal
    adj, adj_high, features, labels = full_load_data(dataset)
    labels = torch.nn.functional.one_hot(labels).float()
    
    nnodes = adj.shape[0]
    adj_dense = adj.to_dense()
    adj_dense[adj_dense!=0] = 1
    adj_dense = adj_dense - torch.diag(torch.diag(adj_dense))
    #D = torch.diag(torch.sum(adj_dense,1))

    one = torch.ones(nnodes,nnodes)
    adj_C = one - adj_dense - torch.eye(nnodes)
    #total_degree = torch.sum(D)
    
    feature_dist = torch.cdist(features,features, p=2)
    label_dist = torch.cdist(labels,labels, p=1)
    
    connected_label_dist = label_dist[adj_dense == 1].numpy()
    disconnected_label_dist = label_dist[adj_C == 1].numpy()
    test_label = test.ttest_ind(connected_label_dist, disconnected_label_dist, equal_var=False)
    
    connected_feature_dist = feature_dist[adj_dense == 1].numpy()
    disconnected_feature_dist = feature_dist[adj_C == 1].numpy()
    test_feature = test.ks_2samp(connected_feature_dist, disconnected_feature_dist)
    
    print("Label test: ", test_label, "feature test: ", test_feature)


dataset_stats(args.dataset_name,aggregated = 0)