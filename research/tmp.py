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
from community_pooling import get_preloaded_cluster, community_pooling
from DataSet import HDF5DataSet, PreCluster
from NeuralNet import NeuralNet
# from ginet import GINet
from spatiallinegraph import LineGraph







class GINetConvLayer(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 number_edge_features=1,
                 num_angle=7,
                 bias=False):

        super(GINetConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_angle=num_angle
        self.edge_fdim=in_channels*2+1

        self.fc = nn.Linear(
            self.in_channels, self.out_channels, bias=bias)
        self.fc_edge_attr = nn.Linear(
            2 * self.in_channels + number_edge_features, self.in_channels, bias=bias)
        # self.fc_attention = nn.Linear(
        #     2 * self.out_channels + self.in_channels, 1, bias=bias)
        self.message = nn.Linear((num_angle)*(self.edge_fdim*2), self.out_channels)
        # self.fc_relation=nn.Linear(
        #     self.in_channels*7, self.out_channels, bias=bias)

        self.linear=nn.Linear((num_angle)*2, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):

        size = self.in_channels
        uniform(size, self.fc.weight)
        # uniform(size, self.fc_attention.weight)
        uniform(size, self.fc_edge_attr.weight)

    
    def forward(self, x, edge_index, edge_attr,pos):

        row, col = edge_index
        num_node = len(x)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        # xcol = self.fc(x[col])
        # xrow = self.fc(x[row])
        xcol = x[col]
        xrow = x[row]

        
        # create edge feature by concatenating node feature
        # alpha = torch.cat([xrow, xcol, ed], dim=1) this 表現edge
        # alpha = self.fc_attention(alpha) edgeごとの重みが１つの値ででる
        # alpha = F.leaky_relu(alpha)

        # print(edge_attr.size(),row.size())
        edge_f= torch.cat([xrow, xcol, edge_attr], dim=1)
        # print(edge_f.size())
        # ed = self.fc_edge_attr(edge_f)
        #ここでエッジ更新
        linegraph=LineGraph(edge_index,edge_f,pos,num_nodes=num_node,input_dim=self.in_channels*2+1, output_dim=self.out_channels)
        linegraph.get_graph(edge_index,edge_f)
        message=linegraph.message().t()
        print(message.size())
        print(self.num_angle*(self.edge_fdim*2))
        # message=message.view(len(edge_index[0]),self.num_angle*2)
        output = self.message(message)

        # print(output.size())
        # output
        # output=0
        
        #ここまでエッジ更新
        num_relation=1
        edge_index_2=0
        edge_hidden =output
        # print(num_node)
        assert edge_index.max() < num_node

        node_out = edge_index[1] 
        # + edge_index_2
        node_out=node_out.view(1,-1)
        node_out=node_out.expand(self.out_channels,-1).t()
        # print(node_out.size(),edge_hidden.size())
        # print(node_out)
        out=torch.zeros(num_node,self.out_channels)
        update = scatter_add(edge_hidden , node_out, dim=0,out=out)
        # print(update.size())
        # torch.einsum("ij,jk->",edge_hidden,node_out)
        # update = update.view(num_node, num_relation * edge_hidden.shape[1])
        # print(update.size())
        # update = self.linear(update)
        # update = self.layers[i].activation(update)
        # hidden = hidden + update
        # edge_input = edge_hidden
        
        return update

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
        # print(data.pos)
        data.x = act(self.conv1(
            data.x, data.edge_index, data.edge_attr,data.pos))
        # cluster = get_preloaded_cluster(data.cluster0, data.batch)
        # data = community_pooling(cluster, data)

        # second conv block
        data.x = act(self.conv2(
            data.x, data.edge_index, data.edge_attr,data.pos))
        # cluster = get_preloaded_cluster(data.cluster1, data.batch)
        x, batch = data.x, data.batch

        # INTERNAL INTERACTION GRAPH
        # first conv block
        data_ext.x = act(self.conv1_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr,data.pos))
        # cluster = get_preloaded_cluster(
        #     data_ext.cluster0, data_ext.batch)
        # data_ext = community_pooling(cluster, data_ext)

        # second conv block
        data_ext.x = act(self.conv2_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr,data.pos))
        # cluster = get_preloaded_cluster(
        #     data_ext.cluster1, data_ext.batch)
        x_ext, batch_ext = data_ext.x, data_ext.batch

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
pos=['pos']
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
               pos=pos
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
