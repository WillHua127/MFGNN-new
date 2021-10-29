import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from gnn import GCNConv
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
from tqdm import tqdm



parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_train', type=int, default=1000)
parser.add_argument('--batch_test', type=int, default=10000)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--dataset', type=str, help='Dataset name.', default = 'proteins')
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name = "ogbn-"+args.dataset, root = 'torch_geometric_data/')#, transform=T.ToSparseTensor())
#dataset = PygNodePropPredDataset(name='ogbn-products',
#                                 transform=T.ToSparseTensor())
data = dataset[0]

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']

data.x = scatter(data.edge_attr, data.edge_index[0], dim=0, dim_size=data.num_nodes, reduce='mean').to(device)
#data.adj_t.set_value_(None)


train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                           sizes=[10, 10, 10], batch_size=args.batch_train,
                           shuffle=True, num_workers=args.num_workers)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=args.batch_test, shuffle=False,
                                  num_workers=args.num_workers)

train_idx = train_idx.to(device)

evaluator = Evaluator(name='ogbn-'+args.dataset)

    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index, edge_attr)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index, edge_attr)
        return x
    
    
    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all



def train(model, epoch, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        
        loss = criterion(out.cpu(), data.y[n_id[:batch_size]].to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        pbar.update(batch_size)
        
    pbar.close()
    loss = total_loss / len(train_loader)
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, data.edge_attr)[train_idx]
#     loss = criterion(out, data.y[train_idx].to(torch.float))
#     loss.backward()
#     optimizer.step()

    return loss


@torch.no_grad()
def test(model):
    model.eval()

    y_pred = model.inference(data.x).cpu()
    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    model = GCN(data.num_features, args.hidden_channels, 112,
                args.num_layers, args.dropout).to(device)
    
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, epoch, optimizer)
            result = test(model)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')



if __name__ == "__main__":
    main()
