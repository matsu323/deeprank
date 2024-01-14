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
        self.number_edge_features =number_edge_features
        self.num_angle=num_angle
        self.fc = nn.Linear(
            self.in_channels, self.out_channels, bias=bias)
        self.fc_edge_attr = nn.Linear(
            2 * self.in_channels + number_edge_features, self.in_channels, bias=bias)
        # self.fc_attention = nn.Linear(
        #     2 * self.out_channels + self.in_channels, 1, bias=bias)
        self.message = nn.Linear((self.num_angle)*(self.edge_fdim*2), self.out_channels)
        # self.fc_relation=nn.Linear(
        #     self.in_channels*7, self.out_channels, bias=bias)

        self.linear=nn.Linear((self.num_angle)*2, self.out_channels)
        # print(self.in_channels)
        # self.reset_parameters()

    def reset_parameters(self):

        size = self.in_channels
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.message.weight)
        nn.init.zeros_(self.fc_edge_attr.weight)
        # uniform(size, self.fc.weight)
        # # uniform(size, self.fc_attention.weight)
        # uniform( 2 * self.in_channels + self.number_edge_features, self.fc_edge_attr.weight)
        # uniform((self.num_angle)*(self.edge_fdim*2), self.message.weight)
        # uniform((self.num_angle)*2, self.linear.weight)
        
    
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
        linegraph=LineGraph(edge_index,edge_f,pos,num_nodes=num_node,input_dim=self.in_channels*2+1, output_dim=self.out_channels,num_angle_bin=self.num_angle+1)
        linegraph.get_graph(edge_index,edge_f)
        #message= 7*edgeattr , num edge
        message=linegraph.message()
        # print(message.size(),message.t().size(),xcol.size(),(self.num_angle)*(self.edge_fdim*2),self.in_channels)
        #ここ？
        # print(message.size(),(self.num_angle)*(self.edge_fdim*2))
        # print(self.num_angle*(self.edge_fdim*2))
        # message=message.view(len(edge_index[0]),self.num_angle*2)
        message=message.t()
        output = self.message(message)

        # print(output.size())
        # output
        # output=0
        
        #ここまでエッジ更新
        num_relation=1
        edge_index_2=0
        # print(11)
        edge_hidden =output
        # print(num_node)
        assert edge_index.max() < num_node

        node_out = edge_index[1]
        # + edge_index_2
        node_out=node_out.view(1,-1)
        node_out=node_out.expand(self.out_channels,-1).t()
        # print(node_out.size(),edge_hidden.size())
        # print(node_out)
        out=torch.zeros(num_node,self.out_channels).to('cuda')
        update = scatter_add(edge_hidden , node_out, dim=0,out=out)
        # print(update.size())
        # torch.einsum("ij,jk->",edge_hidden,node_out)
        # update = update.view(num_node, num_relation * edge_hidden.shape[1])
        # print(update.size())
        # update = self.linear(update)
        # update = self.layers[i].activation(update)
        # hidden = hidden + update
        # edge_input = edge_hidden
        # print(update.size())
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
        #ここ
        act = F.relu
        data_ext = data.clone()

        # EXTERNAL INTERACTION GRAPH
        # first conv block
        # print(data.pos)
        # print(data.x.size())
        
        # intdata= act(self.conv1(
        #     data.x, data.edge_index, data.edge_attr,data.pos))
       
        
        # # second conv block
        # intdata= act(self.conv2(
        #     intdata, data.edge_index, data.edge_attr,data.pos))
        # x, batch = intdata, data.batch

        # # INTERNAL INTERACTION GRAPH
        # # first conv block
        # extdata = act(self.conv1_ext(
        #     data_ext.x, data_ext.edge_index, data_ext.edge_attr,data.pos))

        # # second conv block
        # extdata = act(self.conv2_ext(
        #     extdata, data_ext.edge_index, data_ext.edge_attr,data.pos))

        # x_ext, batch_ext = extdata, data_ext.batch

        # # FC
        # x = scatter_mean(x, batch, dim=0)
        # x_ext = scatter_mean(x_ext, batch_ext, dim=0)

        # x = torch.cat([x, x_ext], dim=1)
        # x = act(self.fc1(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.fc2(x)
        # print(data,data.batch)
        data.x= act(self.conv1(
            data.x, data.edge_index, data.edge_attr,data.pos))
        # print(data,batch)
        cluster = get_preloaded_cluster(data.cluster0, data.batch)
        # print(data.size())
        data = community_pooling(cluster, data)
        # print(data.size())
        # second conv block
        data.x= act(self.conv2(
            data.x, data.edge_index, data.edge_attr,data.pos))
        cluster = get_preloaded_cluster(data.cluster1, data.batch)
        # x, batch = data.x, data.batch
        # print(data.x.size(),data.batch.size())
        x, batch = max_pool_x(cluster, data.x, data.batch)
        # print(x.size(),batch.size())
       
        # INTERNAL INTERACTION GRAPH
        # first conv block
        data_ext.x = act(self.conv1_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr,data_ext.pos))
        cluster = get_preloaded_cluster(
            data_ext.cluster0, data_ext.batch)
        data_ext = community_pooling(cluster, data_ext)
        # second conv block
        data_ext.x = act(self.conv2_ext(
            data_ext.x, data_ext.edge_index, data_ext.edge_attr,data_ext.pos))
        cluster = get_preloaded_cluster(
            data_ext.cluster1, data_ext.batch)
        x_ext, batch_ext = max_pool_x(
            cluster, data_ext.x, data_ext.batch)

        # x_ext, batch_ext = data_ext.x, data_ext.batch
        # print(data_ext.cluster1,data_ext.x.size(),data_ext.batch.size())
        # x_ext, batch_ext = max_pool_x(data_ext.cluster1, data_ext.x, data_ext.batch)
        # print(x_ext.size(),batch_ext.size())
        # FC
        x = scatter_mean(x, batch, dim=0)
        x_ext = scatter_mean(x_ext, batch_ext, dim=0)

        x = torch.cat([x, x_ext], dim=1)
        x = act(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)

        return x
    

# model.plot_loss(name='plot_loss')

# train_metrics = model.get_metrics('train')
# print('training set - accuracy:', train_metrics.accuracy)
# print('training set - sensitivity:', train_metrics.sensitivity)

# eval_metrics = model.get_metrics('eval')
# print('evaluation set - accuracy:', eval_metrics.accuracy)
# print('evaluation set - sensitivity:', eval_metrics.sensitivity)
