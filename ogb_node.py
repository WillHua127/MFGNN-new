import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.base import DGLError
from dgl.transform import reverse
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock


class DGLGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 rank_dim,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=True):
        super(DGLGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._rank_dim = rank_dim
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.bn = nn.BatchNorm1d(rank_dim)

        if weight:
            self.w1 = nn.Parameter(th.Tensor(in_feats, out_feats))
            self.w2 = nn.Parameter(th.Tensor(in_feats+1, rank_dim))
            self.v = nn.Parameter(th.Tensor(rank_dim, out_feats))
            #self.weight_sum = nn.Parameter(th.Tensor(in_feats, out_feats))
            #self.weight2 = nn.Parameter(th.Tensor(rank_dim, out_feats))
            #self.bias = nn.Parameter(th.Tensor(rank_dim))
        else:
            self.register_parameter('weight', None)
            


        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.v)
    
    def _elementwise_product(self, nodes):
        return {'h_prod':th.prod(nodes.mailbox['m_prod'],dim=1)}
    
    def _elementwise_sum(self, nodes):
        return {'h_sum':th.sum(nodes.mailbox['m_sum'],dim=1)}


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
            #aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            feat_sumsrc = th.matmul(feat_src, self.w1)
            feat_prodsrc = th.tanh(th.matmul(th.cat((feat_src, th.ones([feat_src.shape[0],1]).to('cuda:0')),1), self.w2))
            graph.srcdata['h_sum'] = feat_sumsrc
            graph.srcdata['h_prod'] = feat_prodsrc
            graph.update_all(fn.copy_src('h_sum', 'm_sum'), self._elementwise_sum)
            graph.update_all(fn.copy_src('h_prod', 'm_prod'), self._elementwise_product)
            
            rst = graph.dstdata['h_sum'] + th.matmul(graph.dstdata['h_prod'], self.v)


            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            #if self.bias is not None:
                #rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

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
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
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

def compute_acc(pred, labels, evaluator, dataset):
    """
    Compute the accuracy of prediction given the labels.
    """
    if dataset in {"proteins"}:
        acc = evaluator.eval({
            'y_true': labels,
            'y_pred': pred,
        })['rocauc']
    else:
        acc = evaluator.eval({
            'y_true': labels.unsqueeze(1),
            'y_pred': th.argmax(pred, dim=1).unsqueeze(1),
        })['acc']
    #return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    return acc


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device, evaluator, dataset):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid], evaluator, dataset), compute_acc(pred[test_nid], labels[test_nid], evaluator, dataset), pred

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

def convert_mag_to_homograph(g, device):
    """
    Featurize node types that don't have input features (i.e. author,
    institution, field_of_study) by averaging their neighbor features.
    Then convert the graph to a undirected homogeneous graph.
    """
    src_writes, dst_writes = g.all_edges(etype="writes")
    src_topic, dst_topic = g.all_edges(etype="has_topic")
    src_aff, dst_aff = g.all_edges(etype="affiliated_with")
    new_g = dgl.heterograph({
        ("paper", "written", "author"): (dst_writes, src_writes),
        ("paper", "has_topic", "field"): (src_topic, dst_topic),
        ("author", "aff", "inst"): (src_aff, dst_aff)
    })
    new_g = new_g.to(device)
    new_g.nodes["paper"].data["feat"] = g.nodes["paper"].data["feat"]
    new_g["written"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["has_topic"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["aff"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    g.nodes["author"].data["feat"] = new_g.nodes["author"].data["feat"]
    g.nodes["institution"].data["feat"] = new_g.nodes["inst"].data["feat"]
    g.nodes["field_of_study"].data["feat"] = new_g.nodes["field"].data["feat"]

    # Convert to homogeneous graph
    # Get DGL type id for paper type
    target_type_id = g.get_ntype_id("paper")
    g = dgl.to_homogeneous(g, ndata=["feat"])
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    # Mask for paper nodes
    g.ndata["target_mask"] = g.ndata[dgl.NTYPE] == target_type_id
    return g


#### Entry point
def run(args, device, data, evaluator, dataset):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SampleCPPooling(in_feats, args.num_hidden, args.rank, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if dataset in {'proteins'}:
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(nfeat, labels, seeds, input_nodes)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels, evaluator, dataset)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, device, evaluator, dataset)
            if args.save_pred:
                np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            print('Eval Acc {:.4f}'.format(eval_acc))
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    return best_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=300)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='25,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--dataset', type=str,
                    help='Dataset name.', default = 'arxiv')
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--rank', type=int, default=256)
    args = argparser.parse_args()
    
    args.cuda = not args.no_cuda and th.cuda.is_available()
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load ogbn-products data
    #data = DglNodePropPredDataset(name='ogbn-products')
    data = DglNodePropPredDataset(name = "ogbn-"+args.dataset, root = 'torch_geometric_data/')
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
#     if args.cuda:
#         train_idx = train_idx.cuda()
#         val_idx = val_idx.cuda()
#         test_idx = test_idx.cuda()
    graph, labels = data[0]
    n_classes = (labels.max() + 1).item()
    #graph = graph.to(device)
    if args.dataset == "arxiv":
        graph = dgl.add_reverse_edges(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)
        graph.ndata['feat'] = graph.ndata['feat'].float()
        labels = labels[:, 0].to(device)
    elif args.dataset == "mag":
        labels = labels["paper"]
        train_idx = train_idx["paper"]
        val_idx = val_idx["paper"]
        test_idx = test_idx["paper"]
        g = convert_mag_to_homograph(g, device)
        labels = labels[:, 0].to(device)
    elif args.dataset == "proteins":
        n_classes = labels.shape[1]
        graph.update_all(fn.copy_e("feat","feat_copy"),fn.sum("feat_copy","feat"))
        #one_hot = th.zeros(graph.number_of_nodes(), n_classes)
        #one_hot[train_idx, labels[train_idx,0]]=1
        #graph.ndata['feat'] = th.cat([graph.ndata['feat'], one_hot],dim=1)
        graph.ndata['feat'] = graph.ndata['feat'].float()
        labels = labels[:,].float().to(device)
    else:
        graph = dgl.add_self_loop(graph)
        graph.ndata['feat'] = graph.ndata['feat'].float()
        labels = labels[:, 0].to(device)

    nfeat = graph.ndata.pop('feat').to(device)
    

    in_feats = nfeat.shape[1]
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph
    evaluator = Evaluator(name='ogbn-'+args.dataset)

    # Run 10 times
    test_accs = []
    #dropout = [0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9]
    for i in range(1):
        test_accs.append(run(args, device, data, evaluator, args.dataset))
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
        print('Max test acc/acc:', np.max(test_accs), ',',test_accs[-1])
        print('hidden/dropout/wd/rank:', args.num_hidden, ',', args.dropout,',',args.wd,',',args.rank)


