import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
import time
import itertools

from util_gin import load_data, separate_data
from graphcnn import GraphCNN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

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

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=1,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--rank', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="average", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    
    
    
    patience = 100
    best_result = 0
    best_std = 0
    best_dropout = None
    best_weight_decay = None
    best_lr = None
    best_time = 0
    best_epoch = 0

    #lr = [0.05, 0.01,0.002]#,0.01,
    #weight_decay = [1e-4]#,5e-4,5e-5, 5e-3] #5e-5,1e-4,5e-4,1e-3,5e-3
    dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]
    #for args.lr, args.dropout in itertools.product(lr, dropout):
    for args.dropout in itertools.product(dropout):
        result = np.zeros(10)
        t_total = time.time()
        num_epoch = 0
        for idx in range(10):

            ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
            train_graphs, test_graphs = separate_data(graphs, args.seed, idx)

            model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, args.rank, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            tlss_mn = np.inf
            tacc_mx = 0.0
            curr_step = 0
            best_test = 0


            for epoch in range(1, args.epochs + 1):
                num_epoch = num_epoch+1
                #scheduler.step()

                avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
                acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)
                scheduler.step()

                if not args.filename == "":
                    with open(args.filename, 'w') as f:
                        f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                        f.write("\n")
                print("")

                if acc_train >= tacc_mx or avg_loss <= tlss_mn:
                    if acc_train >= tacc_mx and avg_loss <= tlss_mn:
                        best_test = acc_test
                        best_training_loss = avg_loss
                    vacc_mx = np.max((acc_train, tacc_mx))
                    vlss_mn = np.min((avg_loss, tlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience or torch.isnan(avg_loss):
                        break

                #print(model.eps)
            result[idx] = best_test
            del model, optimizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
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
