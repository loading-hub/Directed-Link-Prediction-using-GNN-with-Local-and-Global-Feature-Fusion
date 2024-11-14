import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import multiprocessing as mp
import torch

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label#图的标签
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())#边的度

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(x, y): z for (x, y), z in list(edge_features.items())}#如果边有特征，则以字典的形式存储在edge_features

            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):#有向图在全图范围内取负样本完成调整
    # get upper triangular matrix有向网络里不能只取上三角矩阵，要去全部
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i !=j and net[i, j] == 0:#i < j and不要，有向网络里可能i>j,从全图范围内取样本
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)#完成
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        '''print("Time eplased for subgraph extraction: {}s".format(end-start))
        print('g_list')
        print(g_list[0].edge_features)
        quit()'''
        return g_list
        

    print('Enclosing subgraph extraction begins...')
    #使用helper进行子图提取
    train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)
    
def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):#提取子图
    # extract the h-hop enclosing subgraph around link 'ind'要预测的边
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)#fringe为邻居节点

        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:#设置每跳最大点个数
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)

    subgraph = A[nodes, :][:, nodes]#返回包含nodes里的数字的任意组合的边（如果存在）nodes=[0,1],返回(0,0),(0,1),(1,1),(1,0)以及对应的权重
    # apply node-labeling
    labels = node_label(subgraph)#对子图的点进行标记
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):#找邻居节点时可以用无向图，找到出和入的边
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])#通过出去的边找邻居

        _,nei2,_ = ssp.find(A[node, :])#通过进来的边找邻居

        nei = np.append(nei, nei2)

        nei = set(nei)

        res = res.union(nei)
    return res

def ComputeCore(G, k):
    G_k = nx.k_core(G, k) #用的是networkx中的方法，直接返回k-core
    return G_k

def node_label(subgraph):#有用，但作用有限，需调整.把点进行标记
    # an implementation of the proposed double-radius node labeling (DRNL)

    #quit()
    import networkx as nx
    import numpy as np
    from scipy.sparse import csc_matrix
    G = nx.Graph()
    G = G.to_directed()
    sub= csc_matrix(subgraph).toarray()
    nodes = range(sub.shape[0])
    G.add_nodes_from(nodes)

    for i in range(sub.shape[0]):
        for j in range(sub.shape[0]):
            if (sub[i][j] == 1):
                G.add_edge(i, j)
    degree=G.degree()#去掉前2个，放进数组
    kcore=[]
    for i in range(sub.shape[0]):
        kcore.append(0)
    for i in range(1,sub.shape[0]+1):

        KG = ComputeCore(G, i)
        if KG.size()!=0:
            for j in KG.nodes:
                kcore[j] = kcore[j] + 1

    kcore=kcore[2:]
    nkcore=[]
    maxk=0
    for i in kcore:
        if i>maxk:
            maxk=i
    #print(kcore)
    for i in kcore:
        nkcore.append(maxk+1-i)
    de=[]
    ff=[]
    f=0
    for i in degree:
        if(f>=2):
            de.append(i[1])
        f=f+1
    de=np.array(de).astype(int)
    maxi=0
    for i in de:
        if i>maxi:
            maxi=i
    for i in de:
        ff.append(int(i/maxi))
    ff=np.array(ff).astype(int)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)

    dist_to_0 = dist_to_0[1:, 0]

    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)

    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)


    #labels = (1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 ))*nkcore#
    #labels = (1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2))#cora-auc: 0.84996,ap: 0.93332  citeseer-auc: 0.81046,ap: 0.93769


    labels = (1 +dist_to_0.astype(int)+dist_to_1.astype(int)+ np.minimum(dist_to_0, dist_to_1).astype(int))#*nkcore#cora-auc: 0.82840,ap: 0.92158 citeseer-auc: 0.78131,ap: 0.92756
    #labels=1+(dist_to_0+dist_to_1).astype(int)*nkcore #cora-auc: 0.82935,ap: 0.86025  citeseer-auc: 0.78557,ap: 0.86903
    #labels=nkcore


    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    labels[labels <= 0] = 0
    #labels[labels!=0]=0#cora-auc: 0.73146,ap: 0.74218 citeseer auc: 0.58489,ap: 0.54792
    #print(labels)
    return labels

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def single_line(batch_graphs):
    pbar = tqdm(batch_graphs, unit='iteration')
    graphs = []
    for graph in pbar:
        #line_graph, labels = to_line(graph, graph.node_tags)
        line_test(graph, graph.node_tags)
        #graphs.append(line_graph)
    return graphs

def gnn_to_line(batch_graph, max_n_label):
    start = time.time()
    pool = mp.Pool(16)
    #pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(parallel_line_worker, [(graph, max_n_label) for graph in batch_graph])
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    g_list = [g for g in results]
    return g_list

def parallel_line_worker(x):
    return to_line(*x)

def to_line(graph, max_n_label):#没有用到
    edges = graph.edge_pairs
    edge_feas = edge_fea(graph, max_n_label)/2
    edges, feas = to_direct(edges, edge_feas)
    edges = torch.tensor(edges)
    data = Data(edge_index=edges, edge_attr=feas)

    data.num_nodes = graph.num_nodes

    data = LineGraph()(data)
    data.num_nodes = graph.num_edges
    data['y'] = torch.tensor([graph.label])
    return data

def to_edgepairs(graph):
    x, y = zip(*graph.edges())
    num_edges = len(x)
    edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
    edge_pairs[:, 0] = x
    edge_pairs[:, 1] = y
    edge_pairs = edge_pairs.flatten()
    return edge_pairs
#把原图转化为线图，并调整边的特征
def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:




        edges = graph.edge_pairs
        node_fea=graph.node_features
        edge_feas = edge_fea(graph, max_n_label)/2#给点标签进行onehot编码
        edges, feas = to_direct(node_fea,edges, edge_feas)#通过节点的标签转化为边的特征，edges的数据为2个列表，对应位置的index组成一条边，而feas为一堆列表，每一个对应一条边的特征Fquit

        edges = torch.tensor(edges)

        data = Data(edge_index=edges, edge_attr=feas)#输入边的index和边的特征
        data.num_nodes = graph.num_nodes #这一行多余的注释后对算法无影响


        data = LineGraph(force_directed=True)(data)#转化为线图参数=TRUE表示为有向图默认无向图

        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs

def edge_fea(graph, max_n_label):
    node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1,1)
    node_tag.scatter_(1, tags, 1)
    return node_tag# 产生及返回节点的标签

def edge_fea2(labels, edges):
    feas = []
    for i in range(edges.shape[1]):
        fea = [labels[edges[0][i]], labels[edges[1][i]]]
        fea.sort()
        feas.append(fea)
    feas = np.reshape(feas, [-1, 2])
    feas = np.array([feas[:,0], feas[:,1]], dtype=np.float32)
    return torch.tensor(feas/2)
    
def to_undirect2(edges):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    target_edge = np.array([[0,1],[1,0]])
    return np.concatenate([target_edge, sr, rs], axis=1)#无特征的方法
    
def to_direct(node_fea,edges, edge_fea):#把点的特征转换为边的特征
    edges = np.reshape(edges, (-1,2 ))

    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)

    feal=node_fea[sr[0]]
    fear=node_fea[sr[1]]
    feal=torch.cat([torch.tensor(feal), torch.tensor(fear)], 1)
    fea_s = edge_fea[sr[0,:], :]
    #fea_s = fea_s.repeat(2,1)
    fea_r = edge_fea[sr[1,:], :]
    #fea_r = fea_r.repeat(2,1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    fea_body = torch.cat([fea_body, feal], 1)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return sr, fea_body#有特征的方式


def line_test(graph, label):
    edges = graph.edge_pairs
    edges= to_undirect2(edges)
    feas = edge_fea2(label, edges)
    data = Data(edge_index=torch.tensor(edges), edge_attr=feas.T)
    data = LineGraph()(data)
    elist = data['edge_index'].numpy()
    #elist = [(elist[0][i], elist[1][i]) for i in range(len(elist[0]))]
    #nx_graph = nx.Graph()
    #nx_graph.add_edges_from(elist)
    #return nx_graph, data['x'].numpy()
    #return nx
    
    
    
    