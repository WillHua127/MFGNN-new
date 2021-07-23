from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib
import itertools
from tqdm import tqdm

from utils import load_data, accuracy, separate_data, load_gc_data
from models import GIN



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--idx', type=int, default=0,
                    help='Split number.')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default = 'MUTAG')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training and validation (default: 32)')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers (default: 5)')
parser.add_argument('--num_mlp_layers', type=int, default=1,
                    help='number of MLP layers(default: 2). 1 means linear model.')
parser.add_argument('--graph_pooling_type', type=str, default="cp", choices=["sum", "mean", "max", "cp"],
                    help='type of graph pooling: sum, mean or max')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "mean", "max"],
                    help='type of neighboring pooling: sum, mean or max')
parser.add_argument('--learn_eps', action="store_true",
                    help='learn the epsilon weighting')
parser.add_argument('--iters_per_epoch', type=int, default=50,
                    help='number of iterations per each epoch (default: 50)')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    

# Load data
#edge_dict, features, labels, edge_index = full_load_data(args.dataset_name, args.sub_dataname)
graphs, labels, n_classes = load_gc_data(args.dataset_name)
nfeats = len(graphs[0].ndata['attr'][0])
criterion = nn.CrossEntropyLoss()

def train(args, model, train_graphs, train_labels, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        feat = [graph.ndata['attr'] for graph in batch_graph]
        #print(batch_graph[0], feat[0].shape)
        #feat = torch.cat([graph.ndata['attr'] for graph in batch_graph],0)
        label = torch.LongTensor([train_labels[idx] for idx in selected_idx])
        if args.cuda:
            label = label.cuda()
            feat = feat.cuda()
        
        output = model(batch_graph, feat)

        #compute loss
        loss = criterion(output, label)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

def pass_data_iteratively(model, graphs, minibatch_size = 32):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        sampled_graphs = [graphs[j] for j in sampled_idx]
        feat = [graph.ndata['attr'] for graph in sampled_graphs]
        #output.append(model([graphs[j] for j in sampled_idx]).detach())
        output.append(model(sampled_graphs, feat).detach())
    return torch.cat(output, 0)

    
def test(args, model, train_graphs, test_graphs, train_labels, test_labels, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(train_labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(test_labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test
    
def main():
    patience = 50
    best_result = 0
    best_std = 0
    best_dropout = None
    best_weight_decay = None
    best_lr = None
    best_time = 0
    best_epoch = 0

    lr = [0.05, 0.01,0.002]#,0.01,
    weight_decay = [1e-4,5e-4,5e-5, 5e-3] #5e-5,1e-4,5e-4,1e-3,5e-3
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]
    for args.lr, args.dropout in itertools.product(lr, dropout):
        result = np.zeros(10)
        t_total = time.time()
        num_epoch = 0
        for idx in range(10):
            train_graphs, test_graphs, train_labels, test_labels = separate_data(graphs, labels, args.seed, idx)

            model = GIN(args.num_layers, args.num_mlp_layers,
                    nfeats, args.hidden, n_classes,
                    args.dropout, args.learn_eps,
                    args.graph_pooling_type, args.neighbor_pooling_type, args.batch_size)

            if args.cuda:
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            tlss_mn = np.inf
            tacc_mx = 0.0
            curr_step = 0
            best_test = 0

            for epoch in range(1, args.epochs + 1):
                num_epoch = num_epoch+1
                scheduler.step()

                avg_loss = train(args, model, train_graphs, train_labels, optimizer, epoch)
                acc_train, acc_test = test(args, model, train_graphs, test_graphs, train_labels, test_labels, epoch)
                
                if acc_train >= tacc_mx or avg_loss <= tlss_mn:
                    if acc_train >= tacc_mx and avg_loss <= tlss_mn:
                        best_test = acc_test
                        best_training_loss = avg_loss
                    vacc_mx = np.max((acc_train, tacc_mx))
                    vlss_mn = np.min((avg_loss, tlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break
            print("Optimization Finished! Best Test Result: %.4f, Training Loss: %.4f"%(best_test, best_training_loss))
            result[idx] = best_test
            del model, optimizer
            if args.cuda: torch.cuda.empty_cache()
        five_epochtime = time.time() - t_total
        print("Total time elapsed: {:.4f}s, Total Epoch: {:.4f}".format(five_epochtime, num_epoch))
        print("learning rate %.4f, weight decay %.6f, dropout %.4f, Test Result: %.4f"%(args.lr, 0, args.dropout, np.mean(result)))
        if np.mean(result)>best_result:
                best_result = np.mean(result)
                best_std = np.std(result)
                best_dropout = args.dropout
                best_lr = args.lr
                best_time = five_epochtime
                best_epoch = num_epoch

    print("Best learning rate %.4f, Best weight decay %.6f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f, Time/Run: %.4f, Time/Epoch: %.4f"%(best_lr, 0, best_dropout, best_result, best_std, best_time/10, best_time/best_epoch))
    
    

if __name__ == '__main__':
    main()




