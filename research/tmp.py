import sys
import os
import torch
import numpy as np
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
from tqdm import tqdm
import h5py
import copy
from GraphGenMP import GraphHDF5
from tools.CustomizeGraph import add_target
from Graph import Graph
pdb_path = './data/pdb/1ATN/'
pssm_path = './data/pssm/1ATN/'
ref = './data/ref/1ATN/'

# add_target(graph_path='.', target_name='bin_class',
#       target_list='./data/target/1ATN/dummy_target.csv')
# from deeprank_gnn.GraphGenMP import GraphHDF5
# pdb_path = './data/pdb/1ATN/' # path to the docking model in PDB format
# pssm_path = './data/pssm/1ATN/' # path to the pssm files
# GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
#          graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
# import networkx as nx
# file='1ATN_residue.hdf5'
# hdf5 = h5py.File(file,'r+')
# g = Graph()
# g.h52nx(f5name=file , mol='1ATN_10w', molgrp=None)
# molgrp=hdf5['1ATN_10w']
# for e in g.nx.edges:

#     print(g.nx.edges[('A', '443', 'VAL'), ('B', '45', 'LEU')])
    
# print(len(molgrp["nodes"][()]))
# H=nx.line_graph(g.nx)
# cnt=0
# for n in H.nodes:
#     cnt+=1
#     if cnt <2:
#         H.nodes[n]["dist"]=g.nx.edges[n]["dist"]
#         print( n,g.nx.edges[n])
# for e in H.edges:
#         e1,e2=e
#         if e1[0]==e2[0]:
#             nodej=e1[0]
#             nodei=e1[1]
#             nodek=e2[1]
#         elif e1[0]==e2[1]:
#             nodej=e1[0]
#             nodei=e1[1]
#             nodek=e2[0]
#         elif e1[1]==e2[0]:
#             nodej=e1[1]
#             nodei=e1[0]
#             nodek=e2[1]
#         else:
#             nodej=e1[1]
#             nodei=e1[0]
#             nodek=e2[0]
#         posi=g.nx.nodes[nodei]['pos']
#         posj=g.nx.nodes[nodej]['pos']
#         posk=g.nx.nodes[nodek]['pos']
#         vector1=posi-posj
#         vector2=posk-posj
#         i = np.inner(vector1, vector2)
#         n = np.linalg.norm(vector1) * np.linalg.norm(vector2)
#         c = i / n
#         H.edges[e]['angle']=c
#         H.edges[e]['angle_type']=int((c+1)*7//2)
#         print(H.edges[e]['angle_type'])
# # pos = g.nodes[node]['pos']
# print(g.get_score('bin_class'))
# print(pos[0])
# key="bin_class"
# print(molgrp['score/bin_class'][()])

# G = nx.path_graph(4)
# G.add_edges_from((u, v, {"tot": u+v}) for u, v in G.edges)
# G.edges(data=True)
# G=GraphHDF5(pdb_path=pdb_path,graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
# print(G.get_graph(pdb_path=pdb_path,graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4))
# H = nx.line_graph(G._get_one_graph(name, pssm, ref, biopython))
# H.add_nodes_from((node, G.edges[node]) for node in H)
# H.nodes(data=True)

# from torch_scatter import scatter_mean
# #  0 0 4 3 2,1 0   
# #  0,2 4 1,3  0 0 0
# src = torch.Tensor([[2, 0, 1, 4, 3], 
#                     [0, 2, 1, 3, 4]])
# index = torch.tensor([[4, 5, 4, 2, 3], 
#                       [0, 0, 2, 2, 1]])
# out = src.new_zeros((2, 6))

# out = scatter_mean(src, index, out=out, dim=1)

# print(out)
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn

from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_add

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

# torch_geometric import
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import max_pool_x
from torch_geometric.data import DataLoader

# deeprank_gnn import
from deeprank_gnn.community_pooling import get_preloaded_cluster, community_pooling
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.DataSet import HDF5DataSet, PreCluster
from NeuralNet import NeuralNet
# from ginet import GINet






class GINetConvLayer(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 number_edge_features=1,
                 bias=False):

        super(GINetConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(
            self.in_channels, self.out_channels, bias=bias)
        self.fc_edge_attr = nn.Linear(
            number_edge_features, self.in_channels, bias=bias)
        self.fc_attention = nn.Linear(
            2 * self.out_channels + self.in_channels, 1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        size = self.in_channels
        uniform(size, self.fc.weight)
        uniform(size, self.fc_attention.weight)
        uniform(size, self.fc_edge_attr.weight)

    def cal_adj(self,edge_index,message):
        N=torch.max(edge_index).item()
        adjacency_matrix=np.zeros((N+1,N+1))
        rows = edge_index[0]
        cols = edge_index[1]
        # print(adjacency_matrix.shape)
        # calmessage=[i.detach().numpy() for i in message]
        for i in range(len(rows)):
            
            row=rows[i].item()
            col=cols[i].item()
            # print(row,col)
            adjacency_matrix[row][col] = 1
            adjacency_matrix[col][row] = 1
        return adjacency_matrix
    def forward(self, x, edge_index, edge_attr):

        row, col = edge_index
        num_node = len(x)
        edge_attr = edge_attr.unsqueeze(
            -1) if edge_attr.dim() == 1 else edge_attr

        # xcol = self.fc(x[col])
        # xrow = self.fc(x[row])
        xcol = x[col]
        xrow = x[row]

        ed = self.fc_edge_attr(edge_attr)
        # create edge feature by concatenating node feature
        # alpha = torch.cat([xrow, xcol, ed], dim=1)
        # alpha = self.fc_attention(alpha)
        # alpha = F.leaky_relu(alpha)

        # alpha = F.softmax(alpha, dim=1)
        # h = alpha * xcol
        # torch.einsum()

        # torch.cat([row,col])
        out = torch.zeros(num_node, self.in_channels)#out_channnel=16
        message = scatter_add(ed, row, dim=0, out=out)
        message=message+scatter_add(ed, col, dim=0, out=out)#node*inchan
        #col to row
        node_f=scatter_add(xcol, row, dim=0, out=out)
        node_f = node_f+scatter_add(xrow, col, dim=0, out=out)
        message=self.fc(message)
        node_f=self.fc(node_f)
        z=node_f+message
        # adj=self.cal_adj(edge_index,ed)
        # print(edge_index.shape,alpha.shape)
        # print(self.cal_adj(edge_index,ed))

        return z

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels,
                                   self.out_channels)


class GINet(torch.nn.Module):
    # input_shape -> number of node input features
    # output_shape -> number of output value per graph
    # input_shape_edge -> number of edge input features
    def __init__(self, input_shape, output_shape=1, input_shape_edge=1):
        super(GINet, self).__init__()
        self.conv1 = GINetConvLayer(input_shape, 16, input_shape_edge)
        self.conv2 = GINetConvLayer(16, 32, input_shape_edge)

        self.conv1_ext = GINetConvLayer(
            input_shape, 16, input_shape_edge)
        self.conv2_ext = GINetConvLayer(16, 32, input_shape_edge)

        self.fc1 = nn.Linear(2*32, 128)
        self.fc2 = nn.Linear(128, output_shape)
        self.clustering = 'mcl'
        self.dropout = 0.4

    def forward(self, data):
        act = F.relu
        data_ext = data.clone()

        # EXTERNAL INTERACTION GRAPH
        # first conv block
        data.x = act(self.conv1(
            data.x, data.edge_index, data.edge_attr))
        cluster = get_preloaded_cluster(data.cluster0, data.batch)
        data = community_pooling(cluster, data)

        # second conv block
        data.x = act(self.conv2(
            data.x, data.edge_index, data.edge_attr))
        cluster = get_preloaded_cluster(data.cluster1, data.batch)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        # INTERNAL INTERACTION GRAPH
        # first conv block
        data_ext.x = act(self.conv1_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr))
        cluster = get_preloaded_cluster(
            data_ext.cluster0, data_ext.batch)
        data_ext = community_pooling(cluster, data_ext)

        # second conv block
        data_ext.x = act(self.conv2_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr))
        cluster = get_preloaded_cluster(
            data_ext.cluster1, data_ext.batch)
        x_ext, batch_ext = max_pool_x(
            cluster, data_ext.x, data_ext.batch)

        # FC
        x = scatter_mean(x, batch, dim=0)
        x_ext = scatter_mean(x_ext, batch_ext, dim=0)

        x = torch.cat([x, x_ext], dim=1)
        x = act(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)

        return x
    
database = './1ATN_residue.hdf5'

edge_feature=['dist']
node_feature=['type', 'polarity', 'bsa']
            #  'depth', 'hse', 'ic', 'pssm'
target='bin_class'
task='class' 
batch_size=2
shuffle=True
lr=0.001

model = NeuralNet(database, GINet,
               node_feature=node_feature,
               edge_feature=edge_feature,
               target=target,
               index=None,
               task=task, 
               lr=lr,
               batch_size=batch_size,
               shuffle=shuffle,
               )
add_target(graph_path='.', target_name='bin_class',
       target_list='./data/target/1ATN/dummy_target.csv')
model.train(nepoch=1,  hdf5='output.hdf5')
# model.plot_loss(name='plot_loss')

# train_metrics = model.get_metrics('train')
# print('training set - accuracy:', train_metrics.accuracy)
# print('training set - sensitivity:', train_metrics.sensitivity)

# eval_metrics = model.get_metrics('eval')
# print('evaluation set - accuracy:', eval_metrics.accuracy)
# print('evaluation set - sensitivity:', eval_metrics.sensitivity)
