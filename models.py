import torch.nn as nn
import torch.nn.functional as F
from layers import CPlayer, FClayer, GraphConv, GATConv, GINConv
#from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import torch
from torch.nn.parameter import Parameter
import dgl.function as fn
from dgl.utils import expand_as_pair


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)
    

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
        # input projection (no residual)
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0]))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l]))
        # output projection
        #self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1]))
        self.gat_layers.append(FClayer(num_hidden * heads[-2], num_classes))

    def forward(self, g, features):
        h = features
        for i in range(self.num_layers-1):
            h = F.relu(self.gat_layers[i](g, h)).flatten(1)
            h = F.dropout(h, self.dropout, training=self.training)
            
        return self.gat_layers[-1](h)
    

class GCN(nn.Module):
    def __init__(self,
                 in_fea,
                 hidden,
                 out_class,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.n_layers = n_layers
        # input layer
        self.layers.append(GraphConv(in_fea, hidden, activation=F.relu))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden, hidden, activation=F.relu))
        # output layer
        self.layers.append(FClayer(hidden, out_class))

    def forward(self, g, features):
        h = features
        for i in range(self.n_layers-1):
            h = self.layers[i](g, h)
            h = F.dropout(h, self.dropout, training=self.training)
            
        return self.layers[-1](h)
    
class SampleCPPooling(nn.Module):
    def __init__(self, in_fea, hidden, out_class, rank, dropout):
        super(SampleCPPooling, self).__init__()
        #self.cp = CPlayer(in_fea, hidden, 2*hidden)
        #self.fc = FClayer(hidden, out_class)
        #self.cp2 = CPlayer(hidden, out_class, 2)
        self.conv1 = GraphConv(in_fea, hidden, rank)
        self.conv2 = GraphConv(hidden, out_class, rank, out=True)
        self.dropout = dropout

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        out = self.conv2(mfgs[1], (h, h_dst))
        #out = self.fc(h)
        return out
        
class CPPooling(nn.Module):
    def __init__(self, in_fea, hidden, out_class, rank, dropout):
        super(CPPooling, self).__init__()
        self.cp = CPlayer(in_fea, hidden, rank)
        self.fc = FClayer(hidden, out_class)
        self.dropout = dropout

    def forward(self, g, x):
        fea = F.relu(self.cp(g, x))
        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.fc(fea)
        return out
    
class graph_cp_pooling(nn.Module):
    def __init__(self, in_fea, hidden, rank, init=False):
        super(graph_cp_pooling, self).__init__()
        self.W = Parameter(torch.FloatTensor(in_fea, rank))
        self.V = Parameter(torch.FloatTensor(hidden, rank))
        self.init = init
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
                
    
    def _elementwise_product(self, nodes):
        return {'neigh':torch.prod(nodes.mailbox['m'],dim=1)}
    
    
    def forward(self, x):
        if self.init:
            feat = torch.mm(x, self.W)
            feat = torch.prod(feat,0).unsqueeze(0)
            readout = torch.mm(feat, self.V.T)
        else:
            feat = torch.mm(x, self.W)
            feat = torch.prod(feat,0).unsqueeze(0)
            readout = torch.mm(feat, self.V.T)
        return readout

        
class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, batch_size, rank_dim=32):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.graph_pooling_type = graph_pooling_type
        self.batch_size = batch_size

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.cplayers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            if graph_pooling_type == 'cp':    
                self.ginlayers.append(
                    GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps, hidden_dim, rank_dim, output_dim))
            else:
                self.ginlayers.append(
                    GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps, hidden_dim, rank_dim, output_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))
                self.cplayers.append(graph_cp_pooling(input_dim, hidden_dim, rank_dim, init=True))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))
                self.cplayers.append(graph_cp_pooling(hidden_dim, output_dim, rank_dim, init=False))

            

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        elif graph_pooling_type == 'cp':
            self.pool = SumPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        if self.graph_pooling_type == 'cp':
            fea = h
            pooled_one_feas = [0] * len(g)
            #pooled_graph_feas = [torch.cat([self.pool(g[idx], h[idx]) for idx in range(len(g))],0)]
            pooled_graph_feas = [torch.cat([self.cplayers[0](h[idx]) for idx in range(len(g))],0)]
            score_over_layer = 0
            #score_over_layer += self.drop(pooled_graph_feas[0])
            #print(torch.cat([self.cplayers[0](h[idx]) for idx in range(len(g))],0).shape)

            #for i in range(self.num_layers - 1):
                #h = self.ginlayers[i](g, h)
                #fea = [F.relu(self.batch_norms[i](self.ginlayers[i](g[idx], fea[idx]))) for idx in range(len(g))]
                #pooled_fea = torch.cat([self.pool(g[idx], fea[idx]) for idx in range(len(g))],0)
                #pooled_feas.append(pooled_fea)
            #    for idx, graph in enumerate(g):
            #        fea[idx] = F.relu(self.batch_norms[i](self.ginlayers[i](graph, fea[idx])))
                    #print(fea[idx].shape)
                    #fea[idx] = F.relu(self.ginlayers[i](graph, fea[idx]))
                    #print(fea[idx].shape)
                    #print(fea[idx].shape)
                    #fea[idx] = self.ginlayers[i](graph, fea[idx])
            #        pooled_one_feas[idx] = (self.cplayers[i+1](fea[idx]))
                    #print(pooled_one_feas[idx].shape)
            #    pooled_fea = torch.cat(pooled_one_feas,0)
                #pooled_fea = torch.cat(pooled_one_feas,0)
            #    score_over_layer += self.drop(pooled_fea)
                #pooled_graph_feas.append(pooled_fea)

                #fea = torch.cat([self.ginlayers[i](graph, graph.ndata['attr']) for graph in g],0)
                #temp_fea = self.batch_norms[i](torch.cat(fea, 0))
                #temp_fea = F.relu(temp_fea)
                #hidden_rep.append(fea)


            for i, pooled_fea in enumerate(pooled_graph_feas):
                score_over_layer += self.drop(self.linears_prediction[i](pooled_fea))
            #    pooled_h = self.pool(g, h)
            #    score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

            return score_over_layer
        else:
            fea = h
            pooled_one_feas = [0] * len(g)
            pooled_graph_feas = [torch.cat([self.pool(g[idx], h[idx]) for idx in range(len(g))],0)]

            for i in range(self.num_layers - 1):
                #h = self.ginlayers[i](g, h)
                #fea = [F.relu(self.batch_norms[i](self.ginlayers[i](g[idx], fea[idx]))) for idx in range(len(g))]
                #pooled_fea = torch.cat([self.pool(g[idx], fea[idx]) for idx in range(len(g))],0)
                #pooled_feas.append(pooled_fea)
                for idx, graph in enumerate(g):
                    fea[idx] = F.relu(self.batch_norms[i](self.ginlayers[i](graph, fea[idx])))
                    pooled_one_feas[idx] = self.pool(graph, fea[idx])
                pooled_fea = torch.cat(pooled_one_feas,0)
                pooled_graph_feas.append(pooled_fea)

                #fea = torch.cat([self.ginlayers[i](graph, graph.ndata['attr']) for graph in g],0)
                #temp_fea = self.batch_norms[i](torch.cat(fea, 0))
                #temp_fea = F.relu(temp_fea)
                #hidden_rep.append(fea)

            score_over_layer = 0
            for i, pooled_fea in enumerate(pooled_graph_feas):
                score_over_layer += self.drop(self.linears_prediction[i](pooled_fea))
            #    pooled_h = self.pool(g, h)
            #    score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

            return score_over_layer
        
class TwoCPPooling(nn.Module):
    def __init__(self, in_fea, hidden1, hidden2, out_class, rank1, rank2, dropout):
        super(TwoCPPooling, self).__init__()
        self.cp1 = CPlayer(in_fea, hidden1, rank1)
        self.cp2 = CPlayer(hidden1, hidden2, rank2)
        self.fc = FClayer(hidden2, out_class)
        self.dropout = dropout

    def forward(self, g, x):
        fea = F.relu(self.cp1(g, x))
        fea = F.dropout(fea, self.dropout, training=self.training)
        fea = F.relu(self.cp2(g, fea))
        fea = F.dropout(fea, self.dropout, training=self.training)
        out = self.fc(fea)
        return out
    
    
class GCCONV(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GCCONV, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        
    def _elementwise_sum(self, nodes):
        return {'h':torch.sum(nodes.mailbox['m'],dim=1)}


    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            weight = self.weight

            if self._in_feats > self._out_feats:
                if weight is not None:
                    feat_src = torch.tanh(torch.matmul(feat_src, weight))
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, self._elementwise_sum)
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, self._elementwise_sum)
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)
                    

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm
                
                
            if self._activation is not None:
                rst = self._activation(rst)

            return rst
        

class graph_cp_pooling(nn.Module):
    def __init__(self, in_fea, hidden, rank):
        super(graph_cp_pooling, self).__init__()
        self.W = nn.Linear(in_fea, rank)
        self.V = nn.Linear(rank, hidden)

    def forward(self, x):
        fea = self.W(x)
        fea = torch.prod(fea,0).unsqueeze(0)
        #readout = self.V(fea)
        return fea

class GraphCPPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank_dim, output_dim, final_dropout, device):
        super(GraphCPPooling, self).__init__()

        self.final_dropout = final_dropout
        self.device = device

        ###List of MLPs
        self.conv1 = GCCONV(input_dim, hidden_dim)

        self.linears_prediction = torch.nn.ModuleList()
        self.cppools = torch.nn.ModuleList()
        self.linears_prediction.append(nn.Linear(rank_dim, output_dim))
        self.cppools.append(graph_cp_pooling(hidden_dim+1, hidden_dim, rank_dim))


    def forward(self, batch_graph):
        node_feat = [graph.ndata['feat'].to(self.device) for graph in batch_graph]
        edge_feat = [graph.edata['feat'].float().to(self.device) for graph in batch_graph]
        if edge_feat is not None:
            out = [self.conv1(batch_graph[i], node_feat[i], edge_weight=edge_feat[i]) for i in range(len(node_feat))]
        else:
            out = [self.conv1(batch_graph[i], node_feat[i]) for i in range(len(node_feat))]
        hidden_rep_list = [torch.cat([self.cppools[0](torch.hstack([out[idx], torch.ones([out[idx].shape[0],1])])) for idx in range(len(batch_graph))],0)]

        score_over_layer = 0

        for layer, h in enumerate(hidden_rep_list):
            score_over_layer += F.dropout(self.linears_prediction[layer](h), self.final_dropout, training = self.training)

        return score_over_layer  
