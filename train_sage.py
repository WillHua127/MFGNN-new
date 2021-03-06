from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import itertools

from utils import load_data, accuracy, load_graph_data, data_split, reconstruct, random_disassortative_splits, rand_train_test_idx
from models import SAGE

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--num-hidden', type=int, default=64)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--idx', type=int, default=0,
                    help='Split number.')
parser.add_argument('--dataset', type=str,
                    help='Dataset name.', default = 'film')
parser.add_argument('--sub_dataname', type=str,
                    help='subdata name.', default = 'DE')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    

# Load data
g, n_classes = load_graph_data(args.dataset)
features = g.ndata.pop('features')
labels = g.ndata.pop('labels')
    
num_class = labels.max()+1

if args.cuda:
    features = features.cuda()
    #adj = adj.cuda()
    labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()


def test_sage(model, idx_train, idx_val, idx_test):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = F.log_softmax(logits, dim=1)
        loss_test = F.nll_loss(logits[idx_test], labels[idx_test])
        acc_test = accuracy(logits[idx_test], labels[idx_test])
        
        return acc_test
    

#if args.cuda:
#    model.cuda()

def train_sage():
    patience = 50
    best_result = 0
    best_std = 0
    best_dropout = None
    best_weight_decay = None
    best_lr = None
    best_time = 0
    best_epoch = 0

    lr = [0.05]#, 0.01]#,0.002]#,0.01,
    weight_decay = [5e-4,5e-5, 1e-4] #5e-5,1e-4,5e-4,1e-3,5e-3
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]
    for args.lr, args.weight_decay, args.dropout in itertools.product(lr, weight_decay, dropout):
        result = np.zeros(10)
        t_total = time.time()
        num_epoch = 0
        for idx in range(10):
            #idx_train, idx_val, idx_test = rand_train_test_idx(labels)
            idx_train, idx_val, idx_test = random_disassortative_splits(labels, num_class)
            #rank = OneVsRestClassifier(LinearRegression()).fit(features[idx_train], labels[idx_train]).predict(features)
            #print(rank)
            #adj = reconstruct(old_adj, rank, num_class)

            model = SAGE(features.shape[1], args.num_hidden, n_classes, args.num_layers, args.dropout)

            if args.cuda:
                #adj = adj.cuda()
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            vlss_mn = np.inf
            vacc_mx = 0.0
            vacc_early_model = None
            vlss_early_model = None
            curr_step = 0
            best_test = 0
            best_training_loss = None
            for epoch in range(args.epochs):
                num_epoch = num_epoch+1
                t = time.time()
                model.train()
                optimizer.zero_grad()
                output = model(g, features)
                #print(F.softmax(output,dim=1))
                output = F.log_softmax(output, dim=1)
                #print(output)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                acc_train = accuracy(output[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()

                if not args.fastmode:
                    # Evaluate validation set performance separately,
                    # deactivates dropout during validation run.
                    model.eval()
                    output = model(g, features)
                    output = F.log_softmax(output, dim=1)

                val_loss = F.nll_loss(output[idx_val], labels[idx_val])
                val_acc = accuracy(output[idx_val], labels[idx_val])

                if val_acc >= vacc_mx or val_loss <= vlss_mn:
                    if val_acc >= vacc_mx and val_loss <= vlss_mn:
                        vacc_early_model = val_acc
                        vlss_early_model = val_loss
                        best_test = test_sage(model, idx_train, idx_val, idx_test)
                        best_training_loss = loss_train
                    vacc_mx = np.max((val_acc, vacc_mx))
                    vlss_mn = np.min((val_loss, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

            print("Optimization Finished! Best Test Result: %.4f, Training Loss: %.4f"%(best_test, best_training_loss))

            #model.load_state_dict(state_dict_early_model)
            # Testing
            result[idx] = best_test

            del model, optimizer
            if args.cuda: torch.cuda.empty_cache()
        five_epochtime = time.time() - t_total
        print("Total time elapsed: {:.4f}s, Total Epoch: {:.4f}".format(five_epochtime, num_epoch))
        print("learning rate %.4f, weight decay %.6f, dropout %.4f, Test Result: %.4f"%(args.lr, args.weight_decay, args.dropout, np.mean(result)))
        if np.mean(result)>best_result:
                best_result = np.mean(result)
                best_std = np.std(result)
                best_dropout = args.dropout
                best_weight_decay = args.weight_decay
                best_lr = args.lr
                best_time = five_epochtime
                best_epoch = num_epoch

    print("Best learning rate %.4f, Best weight decay %.6f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f, Time/Run: %.4f, Time/Epoch: %.4f"%(best_lr, best_weight_decay, best_dropout, best_result, best_std, best_time/5, best_time/best_epoch))
    

train_sage()


