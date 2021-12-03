from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import scipy
import itertools

from utils import load_data, accuracy, full_load_data, data_split, normalize, normalize_adj,sparse_mx_to_torch_sparse_tensor, preprocess_features, random_disassortative_splits
from models_syn import GCN,TGNN
# from dataset_stats import Svalue


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--param_tunning', action='store_true', default=False,
                    help='Parameter fine-tunning mode')
parser.add_argument('--model_type', type=str,
                    help='Indicate the GNN model we use (gcn,sgc,mlp,acmgcn,acmsgc)', default = 'gcn')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--num_splits', type=int, help='number of training/val/test splits ', default = 10)
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--idx', type=int, default=0,
                    help='Split number.')
parser.add_argument('--base_dataset', type=str,
                    help='base dataset to generate dataset from', default = 'cora')
parser.add_argument('--early_stopping', type=float, default=200,
                    help='early stopping used in GPRGNN')
parser.add_argument('--target_dataset', type=str,
                    help='Indicate if the graph is generating for a given target dataset', default = 'NA')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--record', type=int,help='1 for printing out the recordable results', default = 1)
parser.add_argument("--layers", type=int, default=2, help="number of hidden layers")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.pi = torch.acos(torch.zeros(1)).item() * 2

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Flickr, WikiCS
    from torch_geometric.utils import to_dense_adj, contains_self_loops, remove_self_loops

weight_decay = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3] #5e-5,1e-4,

if args.model_type == 'sgc' : # or args.model_type =='mlp' or args.model_type == 'acmsgc'
    dropout = [0]
else:
    dropout = [0.1, 0.3, 0.5, 0.7, 0.9]
datasets = ['cora']#['chameleon', 'film','squirrel','cora','citeseer','pubmed', 'random']# [args.base_dataset] # ,'chameleon', 'film','squirrel','cora','citeseer', 'random', , 'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics','Amazon_Computers', 'Amazon_Photo', 'random', 'Flickr'  ['chameleon'] #
edge_homos = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] 
# [0.05, 0.1, 0.15, 0.2]
# [0.25, 0.3, 0.35, 0.4] #
# [0.45, 0.5, 0.55, 0.6, 0.65] #
# [0.7, 0.75, 0.8, 0.85, 0.9]
# [0.16, 0.165, 0.17, 0.175]
# [0.18, 0.185, 0.19, 0.195]
# [0.21, 0.22, 0.23, 0.24]
            
#[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1]
 #np.hstack([np.arange(0.005,0.05,0.005),np.arange(0.05,1,0.05)])#np.arange(0.05,1,0.05)
#if True:#args.weight_decay, args.dropout in itertools.product(weight_decay, dropout):
for edge_homo, args.base_dataset in itertools.product(edge_homos, datasets): # np.arange(2), , np.arange(2), np.arange(4) ,[0,2]
    # Load data
    t_total = time.time()
        
    num_edge_same = 4000
    best_result = 0
    best_std = 0
    best_dropout = None
    best_weight_decay = None
    for args.weight_decay, args.dropout in itertools.product(weight_decay, dropout): # if True: #
        result = np.zeros(10)
        #feature_stats = np.zeros([5,10])
        for sample in range(10):
            Path(f"./data_synthesis_regular_graph/{num_edge_same}/{edge_homo}").mkdir(parents=True, exist_ok=True)
            adj = torch.load((f"./data_synthesis_regular_graph/{num_edge_same}/{edge_homo}/adj_{edge_homo}_{sample}.pt")).clone().detach().float()
            labels = (np.argmax(torch.load((f"./data_synthesis_regular_graph/{num_edge_same}/{edge_homo}/label_{edge_homo}_{sample}.pt")).to_dense().clone().detach().float(), axis = 1)).clone().detach() 
            degree = torch.load((f"./data_synthesis_regular_graph/{num_edge_same}/{edge_homo}/degree_{edge_homo}_{sample}.pt")).to_dense().clone().detach().float()
        
            if args.base_dataset in {'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Computers', 'Amazon_Photo'}:
                Path(f"./data_synthesis_regular_graph/features").mkdir(parents=True, exist_ok=True)
                features = torch.tensor(preprocess_features(np.load(("./data_synthesis_regular_graph/features/{}/{}_{}.npy".format(args.base_dataset, args.base_dataset, sample))))).clone().detach()
                
            else:
                Path(f"./data_synthesis_regular_graph/features").mkdir(parents=True, exist_ok=True)
                features = torch.tensor(preprocess_features(torch.load(("./data_synthesis_regular_graph/features/{}/{}_{}.pt".format(args.base_dataset, args.base_dataset, sample))).detach().numpy())).clone().detach()
            #feature_stats[sample, :] = Svalue(adj.to_sparse(), degree, X = features, average_mode=5, labels = labels)
        #print(args.base_dataset, 'Graph Svalue: ', edge_homo, np.mean(feature_stats, 0))
            
            nnodes = adj.shape[0]
            edge = adj.coalesce().indices()
           
            #adj_dense = adj#adj.to_dense() ##
           
            #adj_dense[adj_dense!=0] = 1
            #adj_dense = adj_dense - torch.diag(torch.diag(adj_dense))
            #adj_low = torch.tensor(normalize(adj_dense+torch.eye(nnodes)))            
            #adj_high =  torch.eye(nnodes) - adj_low
            #adj_low = adj_low.to_sparse()
            #adj_high = adj_high.to_sparse()
           
            #print(adj_low)
            if args.cuda:
                features = features.cuda()
                edge = edge.cuda()
                #adj_low = adj_low.cuda()
                #adj_high = adj_high.cuda()
                labels = labels.cuda()
                  
            def test(): #isolated_mask
                model.eval()
                #output = model(features, adj_low, adj_high)
                output = model(features, edge)
                pred = torch.argmax(F.softmax(output,dim=1) , dim=1)
                pred = F.one_hot(pred).float()
                output = F.log_softmax(output, dim=1)
                loss_test = F.nll_loss(output[idx_test], labels[idx_test])
                acc_test = accuracy(output[idx_test], labels[idx_test])
                # acc_test_isolated = accuracy(output[isolated_mask&idx_test], labels[isolated_mask&idx_test])
                # acc_test_connected = accuracy(output[(~isolated_mask)&idx_test], labels[(~isolated_mask)&idx_test])
                # print("Connected Nodes Test Results: %4f, Isolated Nodes Test Results: %4f,"%(acc_test_connected.item(),acc_test_isolated.item()))
                
                #print("Test set results:",
                      #"loss= {:.4f}".format(loss_test.item()),
                     # "accuracy= {:.4f}".format(acc_test.item()))
                return acc_test
            
            
            # Train model
           
            #idx_train, idx_val, idx_test = data_split(idx, args.base_dataset)
            # splits_file_path = 'splits/'+'synthesis'+'_split_0.6_0.2_'+str(sample)+'.npz'
            # with np.load(splits_file_path) as splits_file:
            #     train_mask = splits_file['train_mask']
            #     val_mask = splits_file['val_mask']
            #     test_mask = splits_file['test_mask']
            # idx_train = torch.BoolTensor(train_mask)
            # idx_val = torch.BoolTensor(val_mask)
            # idx_test = torch.BoolTensor(test_mask)
            idx_train, idx_val, idx_test = random_disassortative_splits(labels, labels.max()+1)
            
#             model = GCN(nfeat=features.shape[1],
#                     nhid=args.hidden,
#                     nclass=labels.max().item() + 1,
#                     dropout=args.dropout,
#                     model_type = args.model_type) #, isolated_mask = isolated_mask
            
            model = TGNN(num_layers=args.layers,
                    in_dim=features.shape[1],
                    num_hidden=args.hidden,
                    num_classes=labels.max().item() + 1,
                    dropout=args.dropout)
            if args.cuda:
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
                model.cuda()
                
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_training_loss = None
            best_val_acc  = 0
            best_val_loss = float('inf')
            val_loss_history = torch.zeros(args.epochs)
            best_test = 0
            best_training_loss = None
            for epoch in range(args.epochs):
                t = time.time()
                model.train()
                optimizer.zero_grad()
                #output = model(features, adj_low, adj_high)
                output = model(features, edge)
                output = F.log_softmax(output, dim=1)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                acc_train = accuracy(output[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()
            
                if not args.fastmode:
                    # Evaluate validation set performance separately,
                    # deactivates dropout during validation run.
                    model.eval()
                    #output = model(features, adj_low, adj_high)
                    output = model(features, edge)
                    output = F.log_softmax(output, dim=1)
            
                val_loss = F.nll_loss(output[idx_val], labels[idx_val])
                val_acc = accuracy(output[idx_val], labels[idx_val])
                
                
                if  val_loss < best_val_loss: #:    val_acc > best_val_acc or val_loss < best_val_loss
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_test  = test() #isolated_mask
                    best_training_loss = loss_train
    
                if epoch >= 0:
                    val_loss_history[epoch] = val_loss.detach()
                if args.early_stopping > 0 and epoch > args.early_stopping:
                    tmp = torch.mean(val_loss_history[epoch-args.early_stopping:epoch])
                    if val_loss > tmp:
                        break

            if args.param_tunning:
                print("Optimization of %s for num_edge_same: %.f, edge_homo: %.4f, %s, weight decay %.5f, dropout %.4f, split %d, Best Test Result: %.4f, Training Loss: %.4f"%(args.model_type, num_edge_same, edge_homo, args.base_dataset, args.weight_decay, args.dropout, sample, best_test, best_training_loss)) #torch.norm(torch.mm(torch.transpose(adj_low[idx_train,:], 0, 1),adj_low[idx_train,:])-torch.mm(torch.transpose(adj_low,0,1),adj_low))/torch.norm(torch.mm(torch.transpose(adj_low,0,1),adj_low))
            else:               
                pass
                # print("Optimization for num_edge_same: %.f, edge_homo: %.4f,, %s, split %d, Best Test Result: %.4f, Training Loss: %.4f, Reconstruction Loss:"%(num_edge_same, edge_homo, args.base_dataset, sample, best_test, best_training_loss)) #torch.norm(torch.mm(torch.transpose(adj_low[idx_train,:], 0, 1),adj_low[idx_train,:])-torch.mm(torch.transpose(adj_low,0,1),adj_low))/torch.norm(torch.mm(torch.transpose(adj_low,0,1),adj_low))
            # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
             
            
            #model.load_state_dict(state_dict_early_model)
            # Testing
            result[sample] = best_test
            del model, optimizer
            if args.cuda: torch.cuda.empty_cache()
        if np.mean(result)>best_result:
            best_result = np.mean(result)
            best_std = np.std(result)
            best_dropout = args.dropout
            best_weight_decay = args.weight_decay
        # print("Model Type: %s, num_edge_same: %.f, edge_homo: %.4f, Base Dataset: %s, weight decay %.5f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f"%(args.model_type, num_edge_same, edge_homo, args.base_dataset, args.weight_decay, args.dropout, np.mean(result), np.std(result)))
    if args.record:
        print("Best Result of Model Type: %s, on num_edge_same: %.f, edge_homo: %.4f, Base Dataset: %s, results %.2f %.2f"%(args.model_type, num_edge_same, edge_homo, args.base_dataset, 100*best_result, 100*best_std))
    else:
        print("Best Result of Model Type: %s, on num_edge_same: %.f, edge_homo: %.4f, Base Dataset: %s, weight decay %.5f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f"%(args.model_type, num_edge_same, edge_homo, args.base_dataset, best_weight_decay, best_dropout, best_result, best_std))
 
