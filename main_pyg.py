import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
#import os 

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

#os.environ['CUDA_LAUNCH_BLOCKING'] = 1
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train_batch(args, model, device, train_graphs, optimizer, epoch, task_type):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), desc="Iteration")

    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[int(idx)] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([train_graphs[int(idx)].y for idx in selected_idx]).to(device)


        #backprop
        optimizer.zero_grad()
        #compute loss
        loss = cls_criterion(output.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
        loss.backward()         
        optimizer.step()

        #report
        pbar.set_description('epoch: %d' % (epoch))

def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[int(idx)] for idx in sampled_idx]).detach())
    return torch.cat(output, 0)

def eval_batch(args, model, device, test_graphs, evaluator):
    model.eval()
    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in test_graphs]).to(device)
    result_dict = evaluator.eval({"y_true": labels.unsqueeze(1), "y_pred": pred})

    return result_dict

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch_data = batch.to_data_list().to(device)
        #x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x = [data.x.to(device) for data in batch_data]
        edge_index = [data.edge_index.to(device) for data in batch_data]
        edge_attr = [data.edge_attr.to(device) for data in batch_data]

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            #pred = model(batch)
            pred = model(x, edge_index, edge_attr)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch_data = batch.to_data_list().to(device)
        x = [data.x.to(device) for data in batch_data]
        edge_index = [data.edge_index.to(device) for data in batch_data]
        edge_attr = [data.edge_attr.to(device) for data in batch_data]

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                #pred = model(batch)
                pred = model(x, edge_index, edge_attr)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--iters_per_epoch', type=int, default=1000,
                        help='number of iterations per each epoch (default: 50)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = "ogbg-"+args.dataset, root = 'torch_geometric_data/') 

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(name = "ogbg-"+args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    #train_graphs = dataset[split_idx["train"]]
    #valid_graphs = dataset[split_idx["valid"]]
    #test_graphs = dataset[split_idx["test"]]

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, device=device).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    print(next(model.parameters()).device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)
        #train_batch(args, model, device, train_graphs, optimizer, epoch, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        #train_perf = eval_batch(args, model, device, train_graphs, evaluator)
        #valid_perf = eval_batch(args, model, device, valid_graphs, evaluator)
        #test_perf = eval_batch(args, model, device, test_graphs, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
