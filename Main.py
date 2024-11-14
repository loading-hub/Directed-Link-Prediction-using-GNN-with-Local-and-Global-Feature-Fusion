import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from main1 import *
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net
import argparse
import ctypes
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj, to_undirected, is_undirected
from torch_geometric.nn import GCNConv

import numpy as np
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix

from datasets import get_citation_dataset
from model_digcl import Encoder, Model, drop_feature
from eval_digcl import label_classification
from get_adj import *

import warnings
global nkcore
nkcore=[]
global node_tag

dst='cora_ml'#cora_ml
tr=0.5
warnings.filterwarnings('ignore')
def train(model: Model, x, edge_index):#对比学习
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_weight_1 = cal_fast_appr(
        alpha_1, edge_index, x.shape[0], x.dtype)
    edge_index_2, edge_weight_2 = cal_fast_appr(
        alpha_2, edge_index, x.shape[0], x.dtype)

    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1, edge_weight_1)
    z2 = model(x_2, edge_index_2, edge_weight_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, dataset, x, edge_index, edge_weight, y, final=False):
    model.eval()
    z = model(x, edge_index, edge_weight)
    embeding=z.cpu().detach().numpy()
    #print(torch.cat(((torch.from_numpy(embeding)),(node_tag)),axis=1))#auc: 0.82746,
    np.save('c://embeding.npy', torch.cat(((torch.from_numpy(embeding)),(node_tag)),axis=1))#auc: 0.85443,
    #np.save('d://embeding.npy', embeding)
    label_classification(z, y, data)#z为embeddings，y是点标签
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link Prediction')
    # general settings
    dn=dst+'edge'
    parser.add_argument('--data-name', default=dn, help='network name')
    parser.add_argument('--train-name', default=None, help='train name')
    parser.add_argument('--test-name', default=None, help='test name')
    parser.add_argument('--max-train-num', type=int, default=10000,
                        help='set maximum number of train links (to fit into memory)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-ratio', type=float, default=tr,
                        help='ratio of test links')
    # model settings
    parser.add_argument('--hop', default=3, metavar='S',
                        help='enclosing subgraph hop number, \
                        options: 1, 2,..., "auto"')
    parser.add_argument('--max-nodes-per-hop', default=100,
                        help='if > 0, upper bound the # nodes per hop by subsampling')



    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    if args.hop != 'auto':
        args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)


    '''Prepare data'''
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

    if args.train_name is None:
        args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
        data = sio.loadmat(args.data_dir)
        net = data['net']

        attributes = None
        # check whether net is symmetric (for small nets only)
        if False:
            net_ = net.toarray()
            assert(np.allclose(net_, net_.T, atol=1e-8))
        #Sample train and test links
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num)
    else:
        args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
        args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
        train_idx = np.loadtxt(args.train_dir, dtype=int)
        test_idx = np.loadtxt(args.test_dir, dtype=int)
        max_idx = max(np.max(train_idx), np.max(test_idx))



        net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
        #net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges这是补齐不对称的边，在有向网络里应删掉
        net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
        #Sample negative train and test links
        train_pos = (train_idx[:, 0], train_idx[:, 1])
        test_pos = (test_idx[:, 0], test_idx[:, 1])
        #负采样
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)#采集正负样本


    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()
    #根据正负样本提取2跳子图

    #对A进行对比学习把A替换成对比学习的数据




    parser1 = argparse.ArgumentParser()

    parser1.add_argument('--dataset', type=str, default=dst)  # default='DBLP'
    parser1.add_argument('--gpu_id', type=int, default=0)
    parser1.add_argument('--config', type=str, default='config_digcl.yaml')
    parser1.add_argument('--alpha', type=float, default=0.1)
    parser1.add_argument('--recache', action="store_true",
                        help="clean up the old adj data", default=True)
    parser1.add_argument('--normalize-features',
                        action="store_true", default=True)
    parser1.add_argument('--adj-type', type=str, default='or')
    parser1.add_argument('--curr-type', type=str, default='log')
    args1 = parser1.parse_args()

    assert args1.gpu_id in range(0, 8)
    torch.cuda.set_device(args1.gpu_id)

    config = yaml.load(open(args1.config), Loader=SafeLoader)[args1.dataset]

    torch.manual_seed(config['seed'])
    random.seed(2021)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[
        config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    alpha_1 = 0.1

    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    path = osp.join(osp.expanduser('.'), 'datasets')
    print(args1.normalize_features)



    dataset = get_citation_dataset(
        args1.dataset, args1.alpha, args1.recache, args1.normalize_features, args1.adj_type)


    print("Num of edges ", dataset[0].num_edges)
    print(dataset[0])




    data = dataset[0]

    D=A.tocoo()
    Dx=D.row
    Dy=D.col


    data.edge_index=torch.tensor([Dx,Dy])
    # 把data.edge_index替换成A+1的结构
    # data = A

    G = nx.Graph()
    G = G.to_directed()

    # G.add_nodes_from()
    G.add_nodes_from(range(len(dataset[0].x)))

    for i in range(len(data.edge_index[0])):
        G.add_edge(int(data.edge_index[0][i]), int(data.edge_index[1][i]))


    # nx.draw_networkx(G)
    prall = community_louvain.best_partition(G)
    #prall = nx.pagerank(G, alpha=0.01)  # 自己换成各种标记方法

    nodes = list(G.nodes())
    la = np.array([partition[node] for node in nodes])
    la = torch.FloatTensor(la)
    la = la.reshape(len(la), 1)

    node_tag = la

    node_tag = F.normalize(node_tag, dim=0)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    edge_index_init, edge_weight_init = cal_fast_appr(
        alpha_1, data.edge_index, data.x.shape[0], data.x.dtype)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        a = 0.9
        b = 0.1
        if args1.curr_type == 'linear':
            alpha_2 = a - (a - b) / (num_epochs + 1) * epoch
        elif args1.curr_type == 'exp':
            alpha_2 = a - (a - b) / (np.exp(3) - 1) * \
                      (np.exp(3 * epoch / (num_epochs + 1)) - 1)
        elif args1.curr_type == 'log':
            alpha_2 = a - (a - b) * (1 / 3 * np.log(epoch / (num_epochs + 1) + np.exp(-3)))
        elif args1.curr_type == 'fixed':
            alpha_2 = 0.9
        else:
            print('wrong curr type')
            exit()

        loss = train(model, data.x, data.edge_index)  # x为点的特征，y为对应标签,edge_index为边标签

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    fea1=test(model, dataset, data.x, edge_index_init,
         edge_weight_init, data.y, final=True)  # datdset不参与计算,x为点的特征，y为对应标签





















    import scipy.io

    #fea = scipy.io.loadmat(r'C:\Users\think\Desktop\link prediction\FFD-main\FFD-main\FFD\Python\data\cora_mlfea.mat')['fea']  # 读取mat文件
    fea1 = np.load('c://embeding.npy')
    #fea1=np.load('d://emb.npy')
    #fea1=np.zeros_like(fea1)

    #fea=np.zeros_like(fea1)


    train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop,fea1)#特征在这里输入
    print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))
    dim = len(train_graphs[0].node_features[0])
    #print('feat_dim')
    #print(dim)  #2879*2=5758
    #quit()
    #转化为线图
    train_lines = to_linegraphs(train_graphs, max_n_label)
    test_lines = to_linegraphs(test_graphs, max_n_label)








    # Model configurations

    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = 'gpu'
    cmd_args.num_epochs = 30
    cmd_args.learning_rate = 5e-3
    cmd_args.batch_size = 50
    cmd_args.printAUC = True
    cmd_args.feat_dim = (max_n_label + 1 + dim)*2   #max_n_label + 1+5758           5928
    cmd_args.attr_dim = 0
    #分批次
    train_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=True)
    test_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)

    #分类器设置
    classifier = Net(cmd_args.feat_dim, cmd_args.hidden, cmd_args.latent_dim, cmd_args.dropout)
    if cmd_args.mode == 'gpu':
        classifier = classifier.to("cuda")

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)



    best_auc = 0
    best_auc_acc = 0
    best_acc = 0
    best_acc_auc = 0
    con=''
    #开始训练
    maxauc = 0
    maxap = 0
    for epoch in range(cmd_args.num_epochs):
        classifier.train()
        avg_loss = loop_dataset_gem(classifier, train_loader, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))


        classifier.eval()
        test_loss = loop_dataset_gem(classifier, test_loader, None)

        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print(('average test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f' % (epoch, test_loss[0], test_loss[1], test_loss[2], avg_loss[3])))
        if test_loss[2]>maxauc:
            maxauc=test_loss[2]
        if avg_loss[3]>maxap:
            maxap=avg_loss[3]

        con =con+ str(epoch)+'\t'+str(test_loss[0])+'\t'+str(test_loss[1])+'\t' +str(test_loss[2])+'\t' +str( avg_loss[3])+'\n'



        if best_auc < test_loss[2]:
            best_auc = test_loss[2]
            best_auc_acc = test_loss[3]

        if best_acc < test_loss[3]:
            best_acc = test_loss[3]
            best_acc_auc = test_loss[2]

    #f = open('C:\\Users\\think\\Desktop\\实验数据\\'+dst+str(tr)+' label.txt', mode='w+')
    #f.write(con)
    #f.close()
    print('auc: %.5f,ap: %.5f'%(maxauc,maxap))