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
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax


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
parser.add_argument('--idx', type=int, default=0,
                    help='Split number.')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default = 'cornell')
parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
parser.add_argument("--heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument('--sub_dataname', type=str,
                    help='subdata name.', default = 'DE')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--task', type=str,
                    help='semi-supervised learning or supervised learning.', default = 'sl')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
    
    
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats = self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self.fc_self = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_identity = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_self_high = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        
        #self.att_vec_mlp= nn.Parameter(torch.FloatTensor(size=(1,num_heads, out_feats)))
        #self.att_vec_low= nn.Parameter(torch.FloatTensor(size=(1,num_heads, out_feats)))
        #self.att_vec_high= nn.Parameter(torch.FloatTensor(size=(1,num_heads, out_feats)))
        #self.att_vec = nn.Parameter(torch.FloatTensor(size=(num_heads, 3, 3)))
        #self.att_vec_low = nn.Linear(num_heads, out_feats, 1, bias=False)
        #self.att_vec_high = nn.Linear(num_heads, out_feats, 1, bias=False)
        self.att_vec_mlp= nn.Linear(out_feats * num_heads, 1, bias=False)
        self.att_vec_low = nn.Linear(out_feats * num_heads, 1, bias=False)
        self.att_vec_high = nn.Linear(out_feats * num_heads, 1, bias=False)
        self.att_vec = nn.Linear(3, 3, bias=False)

        
        self.attn_l_low = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r_low = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_l_high = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r_high = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.model_type = 'else'
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_identity.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self_high.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn_l_low, gain=gain)
        nn.init.xavier_uniform_(self.attn_r_low, gain=gain)
        nn.init.xavier_uniform_(self.attn_l_high, gain=gain)
        nn.init.xavier_uniform_(self.attn_r_high, gain=gain)
        #nn.init.xavier_uniform_(self.att_vec_mlp, gain=gain)
        #nn.init.xavier_uniform_(self.att_vec_low, gain=gain)
        #nn.init.xavier_uniform_(self.att_vec_high, gain=gain)
        #nn.init.xavier_uniform_(self.att_vec, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_mlp.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_low.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_high.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec.weight, gain=gain)


    def attention(self, output_low, output_high, output_mlp):
        T = 3
        #print((output_low*self.att_vec_low).shape)
        att = torch.softmax(self.att_vec(torch.sigmoid(torch.cat([self.att_vec_low(output_low) ,self.att_vec_high(output_high) ,self.att_vec_mlp(output_mlp) ],1)))/T,1)
        #att = torch.softmax(self.att_vec(torch.sigmoid(torch.cat([output_low*self.att_vec_low ,output_high*self.att_vec_high ,output_mlp*self.att_vec_mlp ],1)))/T,1)
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None]
    
    def _elementwise_product(self, nodes):
        return {'h_prod':th.prod(nodes.mailbox['m_prod'],dim=1)}
      
    def forward(self, graph, feat):
        with graph.local_scope():
            feat = self.feat_drop(feat)
            if self.model_type == 'acmgat':
                feat_low = (self.fc_self(feat))
                self_low = feat_low.view(-1, self._num_heads, self._out_feats)
                feat_high = (self.fc_self_high(feat))
                self_high = feat_high.view(-1, self._num_heads, self._out_feats)
                
                el_low = (self_low * self.attn_l_low).sum(dim=-1).unsqueeze(-1)
                #print(self_low.shape, self.attn_l_low.shape,(self_low * self.attn_l_low).shape, (self_low * self.attn_l_low).sum(dim=-1).shape, el_low.shape)
                er_low = (self_low * self.attn_r_low).sum(dim=-1).unsqueeze(-1)
                el_high = (self_high * self.attn_l_high).sum(dim=-1).unsqueeze(-1)
                er_high = (self_high * self.attn_r_high).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft_low': self_low, 'el_low': el_low, 'ft_high': self_high, 'el_high': el_high})
                graph.dstdata.update({'er_low': er_low, 'er_high': er_high})
                graph.apply_edges(fn.u_add_v('el_low', 'er_low', 'e_low'))
                graph.apply_edges(fn.u_add_v('el_high', 'er_high', 'e_high'))
                e_low = self.leaky_relu(graph.edata.pop('e_low'))
                e_high = self.leaky_relu(graph.edata.pop('e_high'))
                graph.edata['a_low'] = self.attn_drop(edge_softmax(graph, e_low))
                graph.edata['a_high'] = self.attn_drop(edge_softmax(graph, e_high))
                
                # message passing
                graph.update_all(fn.u_mul_e('ft_high', 'a_high', 'm_high'), fn.sum('m_high', 'ft_high'))
                graph.update_all(fn.u_mul_e('ft_low', 'a_low', 'm_low'), fn.sum('m_low', 'ft_low'))
                #graph.apply_edges(fn.u_mul_e('ft_low', 'a_low', 'm_low'))
                
                #graph.update_all(fn.copy_edge('m_low', 'm'), fn.sum('m', 'ft_low'))
                #graph.apply_edges(fn.u_mul_e('ft_high', 'a_high', 'm_high'))
                #graph.update_all(fn.copy_edge('m_high', 'm'), fn.sum('m', 'ft_high'))
                
                
                low = F.relu(self_low+graph.dstdata['ft_low']).view(feat_low.shape)
                high = F.relu(self_high-graph.dstdata['ft_high']).view(feat_low.shape)
                identity = F.relu(self.fc_identity(feat))#.view(-1, self._num_heads, self._out_feats)
                #print(identity.view(feat_low.shape).reshape(high.shape)==identity)
                #print(identity.view(feat_low.shape).shape)
                att_low, att_high, att_mlp = self.attention(low, high, identity)
                rst = (att_low*low+att_high*high+att_mlp*identity).view(-1, self._num_heads, self._out_feats)
                
            else:
                feat_src = feat_dst = self.fc_self(feat)#.view(-1, self._num_heads, self._out_feats)
                graph.srcdata['h'] = feat_src
                #el = (feat_src * self.attn_l_low).sum(dim=-1).unsqueeze(-1)
                #r = (feat_dst * self.attn_r_low).sum(dim=-1).unsqueeze(-1)
                #graph.srcdata.update({'ft': feat_src, 'el': el})
                #graph.dstdata.update({'er': er})
                #graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                #e = self.leaky_relu(graph.edata.pop('e'))
                # compute softmax
                #graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                # message passing
                #graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                #                 fn.sum('m', 'ft'))
                graph.update_all(fn.copy_src('h', 'm_prod'), self._elementwise_product)
                rst = graph.dstdata['h'].view(-1, self._num_heads, self._out_feats)

            if self.activation:
                rst = self.activation(rst)
            return rst
    
class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 dropout):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.dropout = dropout
        feat_drop = dropout
        attn_drop = 0.0
        negative_slope = 0.2
        self.activation = F.relu
        residual=False
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, residual, self.activation))
        #self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0]))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            #self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l]))
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        #self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes))
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, graph, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](graph, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](graph, h).mean(1)
        return logits

# Load data
#edge_dict, features, labels, edge_index = full_load_data(args.dataset_name, args.sub_dataname)
g,n_classes = load_graph_data(args.dataset_name)
labels = g.ndata.pop('labels')
features = g.ndata.pop('features')
#norm = g.ndata.pop('norm')
heads = ([args.num_heads] * args.layers) + [args.num_out_heads]
    
num_class = labels.max()+1

if args.cuda:
    features = features.cuda()
    device = torch.device('cuda:%d' % 0)
    g = g.to(device)
    labels = labels.cuda()
    #norm = norm.cuda()
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
    for args.lr, args.weight_decay in itertools.product(lr, weight_decay):
        result = np.zeros(10)
        t_total = time.time()
        num_epoch = 0
        for idx in range(10):
            #idx_train, idx_val, idx_test = rand_train_test_idx(labels)
            #idx_train, idx_val, idx_test = random_disassortative_splits(labels, num_class)
            idx_train, idx_val, idx_test = data_split(idx, args.dataset_name)
            #rank = OneVsRestClassifier(LinearRegression()).fit(features[idx_train], labels[idx_train]).predict(features)
            #print(rank)
            #adj = reconstruct(old_adj, rank, num_class)

            model = GAT(
                    num_layers=args.layers,
                    in_dim=features.shape[1],
                    num_hidden=args.hidden,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    dropout=args.dropout)
            #model = TwoCPPooling(in_fea=features.shape[1], out_class=labels.max().item() + 1, hidden1=2*args.hidden, hidden2=args.hidden, dropout=args.dropout)

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
                        best_test = test(model, idx_train, idx_val, idx_test)
                        best_training_loss = loss_train
                    vacc_mx = val_acc
                    vlss_mn = val_loss
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
                #best_dropout = args.dropout
                best_weight_decay = args.weight_decay
                best_lr = args.lr
                best_time = five_epochtime
                best_epoch = num_epoch

    print("Best learning rate %.4f, Best weight decay %.6f, dropout %.4f, Test Mean: %.4f, Test Std: %.4f, Time/Run: %.4f, Time/Epoch: %.4f"%(best_lr, best_weight_decay, 0, best_result, best_std, best_time/5, best_time/best_epoch))
    
    

train_supervised()



