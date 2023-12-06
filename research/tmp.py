import sys
import os
import torch
import numpy as np
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
from tqdm import tqdm
import h5py
import copy

# from deeprank_gnn.GraphGenMP import GraphHDF5
# pdb_path = './data/pdb/1ATN/' # path to the docking model in PDB format
# pssm_path = './data/pssm/1ATN/' # path to the pssm files
# GraphHDF5(pdb_path=pdb_path, pssm_path=pssm_path,
#          graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
from deeprank_gnn.GraphGenMP import GraphHDF5
pdb_path = './data/pdb/1ATN/'
GraphHDF5(pdb_path=pdb_path, 
         graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
fname='./1ATN_residue.hdf5'

remove_file=[]
f = h5py.File(fname, 'r')
print(f.keys())
mol_names = list(f.keys())
if len(mol_names) == 0:
    print('    -> %s is empty ' % fname)
    remove_file.append(fname)
f.close()
