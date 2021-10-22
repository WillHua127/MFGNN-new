import torch
import torch.optim as optim
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from tqdm import tqdm
import argparse
import time
import numpy as np
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
from torch.utils.data import DataLoader
from torch_geometric.datasets import ZINC
import dgl.nn.pytorch as dglnn
import dgl
import os


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def to_dgl(pyg_graph):
    edge = pyg_graph.edge_index
    graph = dgl.DGLGraph((edge[0],edge[1]))
    graph.ndata['feat'] = pyg_graph.x
    return graph, pyg_graph.y

class graph_cp_pooling(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(graph_cp_pooling, self).__init__()
        self.w = torch.nn.Parameter(torch.Tensor(in_feats+1, out_feats))

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w)
        
    def forward(self, graphs):
        #fea = torch.tanh(self.W(x)
        #fea = torch.prod(fea,0).unsqueeze(0)
        return torch.cat([torch.prod(torch.tanh(torch.matmul(torch.cat((g.srcdata['h'], torch.ones([g.srcdata['h'].shape[0],1]).to('cuda:0')),1), self.w)), 0).unsqueeze(0)+torch.sum(g.srcdata['h'], 0).unsqueeze(0) for g in graphs])
    
def readout_nodes(graph, feat, weight=None, op='sum', ntype=None):
    x = feat
    if weight is not None:
        x = x * graph.nodes[ntype].data[weight]
    return dgl.ops.segment.segment_reduce(graph.batch_num_nodes(ntype), x, reducer=op)


### GCN convolution along the graph structure
class DGLGraphConv(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 rank_dim,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(DGLGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._rank_dim = rank_dim
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.w1 = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.w2 = torch.nn.Parameter(torch.Tensor(in_feats+1, rank_dim))
            self.v = torch.nn.Parameter(torch.Tensor(rank_dim, out_feats))
            #self.weight_sum = nn.Parameter(th.Tensor(in_feats, out_feats))
            #self.weight2 = nn.Parameter(th.Tensor(rank_dim, out_feats))
            #self.bias = nn.Parameter(th.Tensor(rank_dim))
        else:
            self.register_parameter('weight', None)
            
        self.bond_encoder = BondEncoder(out_feats)



        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w1)
        torch.nn.init.xavier_uniform_(self.w2)
        torch.nn.init.xavier_uniform_(self.v)
    
    def _elementwise_product(self, nodes):
        return {'h_prod':torch.prod(nodes.mailbox['m_prod'],dim=1)}
    
    def _elementwise_sum(self, nodes):
        return {'h_sum':torch.sum(nodes.mailbox['m_sum'],dim=1)}


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = self.bond_encoder(edge_weight)
                #aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

                
            
            feat_sumsrc = torch.matmul(feat_src, self.w1)
            feat_prodsrc = torch.tanh(torch.matmul(torch.cat((feat_src, torch.ones([feat_src.shape[0],1]).to('cuda:0')),1), self.w2))
            graph.srcdata['h_sum'] = feat_sumsrc
            graph.srcdata['h_prod'] = feat_prodsrc
            #graph.update_all(fn.copy_src('h_prod', 'm_prod'), self._elementwise_product)
            #graph.update_all(fn.copy_src('h_sum', 'm_sum'), self._elementwise_sum)
            graph.update_all(fn.u_mul_e('h_prod', '_edge_weight', 'm_prod'), self._elementwise_product)
            graph.update_all(fn.u_mul_e('h_sum', '_edge_weight', 'm_sum'), self._elementwise_sum)
            rst = graph.dstdata['h_sum'] + torch.matmul(graph.dstdata['h_prod'], self.v)


            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            #if self.bias is not None:
                #rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)
                

            return rst


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, in_dim, emb_dim, rank, drop_ratio = 0.5):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.rank = rank

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        
        #self.convs.append(DGLGraphConv(9, emb_dim, rank, allow_zero_in_degree=True))
        self.convs.append(dglnn.GraphConv(in_dim, emb_dim, allow_zero_in_degree=True))

        for layer in range(num_layer-1):
            #self.convs.append(DGLGraphConv(emb_dim, emb_dim, rank, allow_zero_in_degree=True))
            self.convs.append(dglnn.GraphConv(emb_dim, emb_dim, allow_zero_in_degree=True))
            #self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, graph, x, e=None):
        h = x
        for layer in range(self.num_layer):
            if e is not None:
                h = self.convs[layer](graph, h, edge_weight = e)
            else:
                h = self.convs[layer](graph, h)
            #h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = h
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)


        return h
    
class GNN(torch.nn.Module):

    def __init__(self, num_tasks, in_dim, num_layer = 5, emb_dim = 512, rank=512, drop_ratio = 0.5, graph_pooling = "sum"):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.rank = rank
        self.in_dim = in_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, rank, drop_ratio = drop_ratio)

        self.graph_cp = graph_cp_pooling(self.emb_dim, self.emb_dim)

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, graph, nfeat, efeat=None):
        if efeat is not None:
            h_node = self.gnn_node(graph, nfeat, efeat)
        else:
            h_node = self.gnn_node(graph, nfeat)
            
        #with graph.local_scope():
        #    graph.srcdata['h'] = h_node#torch.cat((h_node, torch.ones([h_node.shape[0],1])),1)
        #    h_graph = self.graph_cp(dgl.unbatch(graph))
        h_graph = readout_nodes(graph, h_node)

        return self.graph_pred_linear(h_graph)
    

def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        graph = batch[0].to(device)
        labels = batch[1].to(device)
        nfeat = graph.ndata['feat'].to(device)

        pred = model(graph, nfeat)
        optimizer.zero_grad()
        is_labeled = labels == labels
        loss = (pred.to(torch.float32)[is_labeled].squeeze() - labels.to(torch.float32)[is_labeled]).abs().mean()
        loss.backward()
        optimizer.step()

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    epoch_test_mae = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        graph = batch[0].to(device)
        labels = batch[1].to(device)
        nfeat = graph.ndata['feat'].to(device)

        with torch.no_grad():
            pred = model(graph, nfeat)
            total_error += (pred.squeeze() - labels).abs().sum().item()
            epoch_test_mae /= (step + 1)
    return epoch_test_mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--rank', type=int, default=512,
                        help='dimensionality of rank units in GNNs (default: 300)')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--wd', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = ZINC(os.path.join('torch_geometric_data','zinc'), subset=True, split='train')
    val_dataset = ZINC(os.path.join('torch_geometric_data','zinc'), subset=True, split='val')
    test_dataset = ZINC(os.path.join('torch_geometric_data','zinc'), subset=True, split='test')
    n_classes = (train_dataset[0].y.max() + 1).item()
    in_feat = train_dataset[0].x.shape[1]


    train_graphs = [to_dgl(g) for g in train_dataset]
    val_graphs = [to_dgl(g) for g in val_dataset]
    test_graphs = [to_dgl(g) for g in test_dataset]
    ### automatic evaluator. takes dataset name as input
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=collate_dgl)
    valid_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=collate_dgl)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=collate_dgl)

    model = GNN(num_tasks = n_classes, in_dim=in_feat, num_layer = args.num_layer, emb_dim = args.emb_dim, rank=args.rank, drop_ratio = args.drop_ratio).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader)
        valid_perf = eval(model, device, valid_loader)
        test_perf = eval(model, device, test_loader)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)

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
