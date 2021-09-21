import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models import GraphCPPooling

from ogb.graphproppred import DglGraphPropPredDataset,collate_dgl
from torch_geometric.data import DataLoader
from dgl.dataloading import GraphDataLoader
import dgl
from ogb.graphproppred import Evaluator


criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [dgl.add_self_loop(train_graphs[idx][0]) for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([train_graphs[idx][1] for idx in selected_idx]).to(device)

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
        output.append(model([dgl.add_self_loop(graphs[j][0]) for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def validation(args, model, evaluator, device, train_graphs, test_graphs, epoch, dataset):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph[1] for graph in train_graphs]).to(device)
    result_dict = evaluator.eval({"y_true": labels.unsqueeze(1), "y_pred": pred})
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))
    if dataset in {"molhiv"}:
        auc_train = result_dict['rocauc']
    elif dataset in {"molpcba"}:
        auc_train = result_dict['ap']
    elif dataset in {"ppa"}:
        auc_train = result_dict['acc']
    elif dataset in {"code2"}:
        auc_train = result_dict['F1']

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph[1] for graph in test_graphs]).to(device)
    result_dict = evaluator.eval({"y_true": labels.unsqueeze(1), "y_pred": pred})
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    if dataset in {"molhiv"}:
        auc_test = result_dict['rocauc']
    elif dataset in {"molpcba"}:
        auc_test = result_dict['ap']
    elif dataset in {"ppa"}:
        auc_test = result_dict['acc']
    elif dataset in {"code2"}:
        auc_test = result_dict['F1']

    print("accuracy train: %f valid: %f" % (acc_train, acc_test))
    print("auc train: %f valid: %f" % (auc_train, auc_test))

    return auc_train, auc_test

def test(args, model, evaluator, device, test_graphs, epoch, dataset):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph[1] for graph in test_graphs]).to(device)
    result_dict = evaluator.eval({"y_true": labels.unsqueeze(1), "y_pred": pred})
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    auc_test = result_dict['rocauc']
    if dataset in {"molhiv"}:
        auc_test = result_dict['rocauc']
    elif dataset in {"molpcba"}:
        auc_test = result_dict['ap']
    elif dataset in {"ppa"}:
        auc_test = result_dict['acc']
    elif dataset in {"code2"}:
        auc_test = result_dict['F1']

    print("accuracy test: %f" % (acc_test))
    print("auc test: %f" % (auc_test))

    return auc_test

def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--hidden_dim', type=int, default=3,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--rank_dim', type=int, default=10,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    dataset = DglGraphPropPredDataset(name = "ogbg-"+args.dataset, root = 'torch_geometric_data/') 
    split_idx = dataset.get_idx_split()
    train_graphs = dataset[split_idx["train"]]
    valid_graphs = dataset[split_idx["valid"]]
    test_graphs = dataset[split_idx["test"]]
    num_classes = (torch.max(torch.LongTensor([dataset[idx][1] for idx in range(len(dataset))]))+1).numpy()

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCPPooling(train_graphs[0][0].ndata['feat'].shape[1], args.hidden_dim, args.rank_dim, num_classes, args.final_dropout,device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    evaluator = Evaluator(name = "ogbg-"+args.dataset)


    patience = 50
    vacc_mx = 0.0
    curr_step = 0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        train_acc, val_acc = validation(args, model, evaluator, device, train_graphs, valid_graphs, epoch, args.dataset)

        if val_acc >= vacc_mx: #or loss_accum <= vlss_mn:
            #if val_acc >= vacc_mx and loss_accum <= vlss_mn:
            curr_step = 0
            best_test = test(args, model, evaluator, device, valid_graphs, epoch, args.dataset)
            vacc_mx = val_acc
            #vlss_mn = np.min((loss_accum, vlss_mn))
            print("Best val: %.4f, best test: %.4f"%(vacc_mx, best_test))
        else:
            curr_step += 1
            if curr_step >= patience:
                break

    

if __name__ == '__main__':
    main()




