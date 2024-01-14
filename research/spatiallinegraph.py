import torch
from torch import nn
# from torchdrug import core, data
import numpy as np
from Graph import Graph
import networkx as nx
import h5py
import math

from torch_geometric.data import Data
# from torch_geometric.data.datapipes import functional_transform
# from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter,scatter_add
from torch_geometric.utils import coalesce, cumsum

# from utils import cumsum,coalesce
class LineGraph(object):
    def __init__(self,edge_index,edge_attr,pos,num_nodes,input_dim, output_dim,x=None, num_angle_bin=8):
        self.num_angle_bin=num_angle_bin
        self.edge_index=edge_index
        self.edge_attr=edge_attr
        self.pos=pos
        self.x=x
        self.num_nodes=num_nodes
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.fea=None
        self.linenum_nodes=None
        
        
    def get_graph(self,edge_index,edge_attr):
        # self.nx = nx.line_graph(origin)
        # self.origin=origin
        
        N = self.num_nodes
        # print(self.edge_index)
        self.edge_index, self.edge_attr = coalesce(self.edge_index, self.edge_attr, num_nodes=N)
        row, col = self.edge_index

        
        i = torch.arange(row.size(0), dtype=torch.long, device='cuda')

        count = scatter(torch.ones_like(row), row, dim=0,
                        dim_size=N, reduce='sum')
        ptr = cumsum(count)

        cols = [i[ptr[col[j]]:ptr[col[j] + 1]] for j in range(col.size(0))]
        rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]

        row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

        self.edge_index = torch.stack([row, col], dim=0)
        self.x = edge_attr
        self.linenum_nodes = self.edge_index.size(1)
        
        self.cal_relation(edge_index)
        # print(self.x)
    # def edge_passing(self):
    #     print(1)
        
    def cal_relation(self,edge_index):
        self.edge_index_easy=edge_index
        node_in, node_out = edge_index
        edge_in, edge_out = self.edge_index
        node_i=node_in[edge_in]
        node_j=node_in[edge_out]
        node_k = node_out[edge_out]

        vector1 = self.pos[node_i] - self.pos[node_j]
        vector2 = self.pos[node_k] - self.pos[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long().clamp(max=self.num_angle_bin - 1)
        self.edge_attr = relation.unsqueeze(-1)
        # print(self.edge_attr)
        # edge_ = torch.cat([edge_list, relation.unsqueeze(-1)], dim=-1)
        return self.edge_attr
    def cal_fea(self):
        
        node_in, node_out = self.edge_index_easy
        edge_in, edge_out = self.edge_index
        node_j=node_in[edge_out]
        self.fea=torch.cat([self.x[edge_in],self.x[edge_out]],dim=1)

        # print(self.x[edge_in].size())
        return self.fea
    # def forward(self):
    #     self.message(self)
    def message(self):
        self.cal_fea()
        
        node_in,node_out = self.edge_index_easy
        edge_in, edge_out = self.edge_index
        # message = self.fea[edge_in]
        # print(message.size())
        # print(self.edge_index.size(),self.edge_attr.size(),len(self.edge_index_easy[0]))
        
        # assert self.edge_index.max() < self.linenum_nodes
        # node_out = self.edge_index.t()*(self.num_angle_bin-1)+self.edge_attr
        
        for i in range(1,self.num_angle_bin):
            
            node_out=torch.cat([self.edge_index,self.edge_attr.reshape(1,-1)],0)
            # print(torch.where(node_out[2:,]==0,torch.zeros(1),torch.ones(1)))
            fea=self.fea_torelation(i,self.fea.t(),node_out)
            node_out=torch.where(node_out[2:,]==i,node_out,torch.zeros_like(node_out))
            # print(node_out.size(),node_out)
            num_fea=self.x.size()[1]*2
            out=torch.zeros(num_fea,len(self.x)).to('cuda')
            # print(node_out.size())
            
            # print(node_out[1:-1],self.x.size()[1]*2,self.x.size())
            # print(self.fea.t().size(),node_out[1:-1].expand(self.input_dim*2,-1).size())
            # print(node_out[1:-1])
            # print(fea.size(),node_out.size(),out.size(),self.input_dim*2,self.x.size())
            #out= 7*edgeattr , num edge
            update= scatter_add(fea,node_out[1:-1].expand(num_fea,-1),dim=1,out=out)
            if i==1:
                output=update
            else:
                output=torch.cat([output,update],0)
            # print(output.size())
            # print(update.size())
        return output
    def fea_torelation(self,i,fea,node_out):
        relation=torch.cat([fea,node_out[2:,]],0)
        relation=torch.where(relation[-1:,]==i,relation,torch.zeros_like(relation))
        # print(fea.size())
        relation=relation[:-1,]
        #print(relation.size())
        return relation

