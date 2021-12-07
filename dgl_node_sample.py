from __future__ import division
from __future__ import print_function
import matplotlib
import itertools

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import dgl.function as fn
import dgl
from dgl.utils import expand_as_pair
import tqdm


from utils import load_data, accuracy, full_load_data, data_split, random_disassortative_splits, rand_train_test_idx, load_graph_data, semi_supervised_splits
#from models import CPPooling, TwoCPPooling, GAT



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--rank', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--idx', type=int, default=0,
                    help='Split number.')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default = 'cornell')
parser.add_argument('--fan-out', type=str, default='5,5')
parser.add_argument("--layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument('--num-workers', type=int, default=0,help="Number of sampling processes. Use 0 for no extra process.")
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--val-batch-size', type=int, default=10000)
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
parser.add_argument('--log-every', type=int, default=50)
parser.add_argument('--eval-every', type=int, default=1)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
device = th.device('cuda:0')
    
class DGLGraphConv(nn.Module):
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
        self._allow_zero_in_degree = allow_zero_in_degree
        self.att1= nn.Linear(out_feats, 1, bias=False)
        self.att2 = nn.Linear(out_feats, 1, bias=False)
        self.att_vec = nn.Linear(2, 2, bias=False)

        self.weight_sum = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_prod = nn.Parameter(torch.Tensor(in_feats, rank_dim))
        self.v = nn.Parameter(torch.Tensor(rank_dim, out_feats))



        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_sum)
        nn.init.xavier_uniform_(self.weight_prod)
        nn.init.xavier_uniform_(self.v)
        self.att1.reset_parameters()
        self.att2.reset_parameters()
        self.att_vec.reset_parameters()

    
    def _elementwise_product(self, nodes):
        return {'h_prod':torch.prod(nodes.mailbox['m_prod'],dim=1)}
    
    def _elementwise_sum(self, nodes):
        return {'h_sum':torch.sum(nodes.mailbox['m_sum'],dim=1)}


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
        
    def attention(self, prod, add):
        T = 2
        att = torch.softmax(self.att_vec(torch.sigmoid(torch.cat([self.att1(prod) ,self.att2(add)],1)))/T,1)
        return att[:,0][:,None],att[:,1][:,None]

    def forward(self, graph, feat):

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

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
#             if self._norm == 'both':
#                 degs = graph.out_degrees().float().clamp(min=1)
#                 norm = torch.pow(degs, -0.5)
#                 shp = norm.shape + (1,) * (feat_src.dim() - 1)
#                 norm = torch.reshape(norm, shp)
#                 feat_src = feat_src * norm


                
            
            feat_sum_src = torch.matmul(feat_src, self.weight_sum)
            feat_prod_src = torch.matmul(feat_src, self.weight_prod)
            #graph.srcdata['h_prod'] = th.tanh(feat_prod_src)#torch.tanh(feat_src)
            graph.srcdata['h_sum'] = feat_sum_src
            graph.srcdata['h_prod'] = torch.tanh(feat_prod_src)
            graph.update_all(fn.copy_src('h_prod', 'm_prod'), self._elementwise_product)
            graph.update_all(fn.copy_src('h_sum', 'm_sum'), self._elementwise_sum)
            #graph.update_all(fn.copy_src('h_sum', 'm_sum'), fn.sum(msg='m_sum', out='h_sum'))
            prod_agg = torch.matmul(graph.dstdata['h_prod'], self.v)
            sum_agg = graph.dstdata['h_sum']
            att_prod, att_sum = self.attention(prod_agg, sum_agg)
            rst = att_prod*prod_agg + att_sum*sum_agg

            #rst = self.batch_norm(rst)
            #print("rst1",rst)
            #print(rst)
            #rst = th.matmul(rst, self.weight2)+graph.dstdata['h_sum']
            #print("rst2",rst)
#             if self._norm != 'none':
#                 degs = graph.in_degrees().float().clamp(min=1)
#                 if self._norm == 'both':
#                     norm = torch.pow(degs, -0.5)
#                 else:
#                     norm = 1.0 / degs
#                 shp = norm.shape + (1,) * (feat_dst.dim() - 1)
#                 norm = torch.reshape(norm, shp)
#                 rst = rst * norm

            #if self.bias is not None:
                #rst = rst + self.bias
                

            return rst


class SampleCPPooling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_rank,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_rank = n_rank
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(DGLGraphConv(in_feats, n_hidden, n_rank))
        for i in range(1, n_layers - 1):
            self.layers.append(DGLGraphConv(n_hidden, n_hidden, n_rank))
        self.layers.append(DGLGraphConv(n_hidden, n_classes, n_rank))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            #sampler = dgl.dataloading.MultiLayerNeighborSampler([int(10)])
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes]
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y

        return y

    
def evaluate(model, g, nfeat, labels, idx_val, idx_test, device):
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nfeat, device)
        pred = F.log_softmax(pred, dim=1)
        val_loss = F.nll_loss(pred[idx_val], labels[idx_val])
        val_acc = accuracy(pred[idx_val], labels[idx_val])
        test_loss = F.nll_loss(pred[idx_test], labels[idx_test])
        test_acc = accuracy(pred[idx_test], labels[idx_test])
    model.train()
    return val_acc,test_acc,val_loss,test_loss


def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

# Load data
#edge_dict, features, labels, edge_index = full_load_data(args.dataset_name, args.sub_dataname)
g,n_classes = load_graph_data(args.dataset_name)
labels = g.ndata.pop('labels')
features = g.ndata.pop('features')
#norm = g.ndata.pop('norm')
g.create_formats_()
    
in_feats = features.shape[1]
num_class = labels.max()+1



if args.cuda:
    features = features.cuda()
    #adj = adj.cuda()
    labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()


    
def test(model, idx_train, idx_val, idx_test):
    model.eval()
    output = model(g, features)
    pred = torch.argmax(F.softmax(output,dim=1) , dim=1)
    pred = F.one_hot(pred).float()
    output = F.log_softmax(output, dim=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test
    
    
def train_supervised():
    patience = 100
    best_result = 0
    best_std = 0
    best_dropout = None
    best_weight_decay = None
    best_lr = None
    best_time = 0
    best_epoch = 0

    lr = [0.05, 0.01,0.002]#,0.01,
    weight_decay = [1e-4,5e-4,5e-5, 5e-3] #5e-5,1e-4,5e-4,1e-3,5e-3
    dropout = [0.1, 0.2, 0.3]#, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]
    for args.lr, args.weight_decay in itertools.product(lr, weight_decay):
        result = np.zeros(10)
        t_total = time.time()
        num_epoch = 0
        for idx in range(10):
            idx_train, idx_val, idx_test = random_disassortative_splits(labels, num_class)
            sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                idx_train,
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)
            model = SampleCPPooling(in_feats, args.hidden, args.rank, n_classes, args.layers, F.relu, args.dropout)

            #model = TwoCPPooling(in_fea=features.shape[1], out_class=labels.max().item() + 1, hidden1=2*args.hidden, hidden2=args.hidden, dropout=args.dropout)

            if args.cuda:
                #adj = adj.cuda()
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            vacc_mx = 0.0
            curr_step = 0
            best_test = 0
            best_training_loss = None
            # Training loop
            avg = 0
            iter_tput = []
        
            
            for epoch in range(args.epochs):
                tic = time.time()
                model.train()
                for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                    tic_step = time.time()
                    # copy block to gpu
                    blocks = [blk.int().to(device) for blk in blocks]

                    # Load the input features as well as output labels
                    batch_inputs, batch_labels = load_subtensor(features, labels, seeds, input_nodes)

                    # Compute loss and prediction
                    optimizer.zero_grad()
                    batch_pred = model(blocks, batch_inputs)
                    batch_pred = F.log_softmax(batch_pred, dim=1)
                    loss_train = F.nll_loss(batch_pred, batch_labels)
                    loss_train.backward()
                    optimizer.step()

                    iter_tput.append(len(seeds) / (time.time() - tic_step))
                    if step % args.log_every == 0:
                        acc = accuracy(batch_pred, batch_labels)
                        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                        print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                            epoch, step, loss_train.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
                toc = time.time()
                if epoch >= 5:
                    avg += toc - tic
                if epoch % args.eval_every == 0:
                    val_acc, test_acc, val_loss, test_loss = evaluate(model, g, features, labels, idx_val, idx_test, device)
                    print('Eval Acc {:.4f}, Eval Loss {:.4f}'.format(val_acc, val_loss))
                    if val_acc >= vacc_mx:# or val_loss <= vlss_mn:
                        curr_step = 0
                        best_test = test_acc
                        vacc_mx = val_acc
                    else:
                        curr_step += 1
                        if curr_step >= patience:
                            break
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(vacc_mx, best_test))
                    
            print("Optimization Finished! Best Test Acc: %.4f, Best Eval Acc: %.4f"%(best_test, vacc_mx))

            #model.load_state_dict(state_dict_early_model)
            # Testing
            result[idx] = best_test

            del model, optimizer
            if args.cuda: torch.cuda.empty_cache()
        print("learning rate %.4f, weight decay %.6f, dropout %.4f, Test Result: %.4f"%(args.lr, args.weight_decay, args.dropout, np.mean(result)))
        if np.mean(result)>best_result:
                best_result = np.mean(result)
                best_std = np.std(result)
                best_dropout = args.dropout
                best_weight_decay = args.weight_decay
                best_lr = args.lr
                best_time = five_epochtime
                best_epoch = num_epoch

    print("Best learning rate %.4f, Best weight decay %.6f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f"%(best_lr, best_weight_decay, best_dropout, best_result, best_std))
    
    

train_supervised()
