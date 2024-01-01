import torch
from torch import nn
# from torchdrug import core, data
import numpy as np
from Graph import Graph
import networkx as nx
class LineGraph(Graph):
    def get_graph(self, origin=Graph,num_angle=8):
        self.nx = nx.line_graph(origin)
        self.origin=origin
        self.get_node_features()
        self.get_edge_features(num_angle)
        return self.nx
    def get_node_features(self):
        for n in self.nx.nodes:
            self.nx.nodes[n]["dist"]=self.origin.edges[n]["dist"]
    def get_edge_features(self,num_angle):
        for e in self.nx.edges:
            e1,e2=e
            if e1[0]==e2[0]:
                nodej=e1[0]
                nodei=e1[1]
                nodek=e2[1]
            elif e1[0]==e2[1]:
                nodej=e1[0]
                nodei=e1[1]
                nodek=e2[0]
            elif e1[1]==e2[0]:
                nodej=e1[1]
                nodei=e1[0]
                nodek=e2[1]
            else:
                nodej=e1[1]
                nodei=e1[0]
                nodek=e2[0]
            posi=self.origin.nx.nodes[nodei]['pos']
            posj=self.origin.nx.nodes[nodej]['pos']
            posk=self.origin.nx.nodes[nodek]['pos']
            vector1=posi-posj
            vector2=posk-posj
            i = np.inner(vector1, vector2)
            n = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            c = i / n
            self.nx.edges[e]['angle']=c
            self.nx.edges[e]['angle_type']=int((c+1)*num_angle//2)
