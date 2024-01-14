from ResidueGraph import ResidueGraph
# pssm="./data/pssm/1ATN_"
import h5py
from tools.CustomizeGraph import add_target
# name="./data/pdb/1ATN/1ATN_1w.pdb"
fname="./1ACB_residue.hdf5"
from GraphGenMP import GraphHDF5
import os
pdb_path = './exsample'
pssm_path = './pssm/pssm_mapped_to_pdb'
# GraphHDF5(pdb_path=pdb_path,pssm_path=pssm_path,biopython=False, graph_type='residue', outfile='tmp.hdf5',
#           nproc=1, tmpdir='./tmpdir')
# # add_target(graph_path='1ATN_residue.hdf5', target_name='fnat',
# #        target_list='./data/target/1ATN/dummy_target.csv')
# add_target(graph_path='./tmp.hdf5', target_name='fnat',
#        target_list='./exsample/fnat.dat')

# f = h5py.File(fname, 'r')
# # print(f["1ACB_cm-it0_9890/score/fnat"][()])
# f.close()
# graph_path='./1ACB_residue.hdf5'
# from Bio.PDB import *
# parser=PDBParser()
# pdbfile="./exsample/1ACB_cm-it0_9890.pdb"
# pdbl = PDBList()
# pdbl.retrieve_pdb_file(pdbfile)
# structure = parser.get_structure('_tmp', pdbfile)
# print(structure[0])


from time import time
# f5 = h5py.File(hdf5, 'a')
graph_path = './1ATN_residue.hdf5'
graph_path=database = './1ACB_residue.hdf5'
# add_target(graph_path=database, target_name='fnat',
#        target_list='./data/target/1ATN/dummy_target.csv')

# f5 = h5py.File(hdf5, 'a')
# if os.path.isfile(graph_path):
#     graphs = [graph_path]
#     for hdf5 in graphs:
#             #   print(hdf5)
#         t=time()
#         f5 = h5py.File(hdf5, 'a')
#         cnt=0
#         j=0
#         for model in f5.keys():
            
#             # cnt+=1
#             if cnt >1000:
#                 print(time()-t)
#                 print(j)
#                 break
#             i=len(f5['{}/edges'.format(model)][()])
#             j=max(i,j)
#             print(f5['{}/score/fnat'.format(model)][()])
#         f5.close()            
          
database = outfile='./1ACB_residue.hdf5'
from tmp import GINet
from NeuralNet import NeuralNet
pdb_path = './400data'
pssm_path = './pssm/pssm_mapped_to_pdb'
outfile ='400data.hdf5'
from time import time
t=time() 
# GraphHDF5(pdb_path=pdb_path,  pssm_path=pssm_path,biopython=False,
#           graph_type='residue', outfile=outfile,
#           nproc=16, tmpdir='./tmpdir')
# print(t-time())
# add_target(graph_path=outfile, target_name='fnat',
#        target_list='./data_in/1ACB/Fnat.dat')


database = outfile
edge_feature=['dist']
node_feature=['type', 'polarity', 'bsa','ic',"charge",
              'pssm']
# node_feature=['type', 'polarity', 'bsa']
t0=time()
pos=['pos']
target='fnat'
task='reg' 
batch_size=128
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
t1=time()
model.train(nepoch=1,  hdf5='output.hdf5')
t = time() - t0
print(t,t1-t0)