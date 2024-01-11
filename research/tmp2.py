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


# if os.path.isfile(graph_path):
#        graphs = [graph_path]
#        for hdf5 in graphs:
#               print(hdf5)
#               f5 = h5py.File(hdf5, 'a')
#               for model in f5.keys():
#                      model_gp = f5['{}'.format(model)]
#                      if 'score' not in model_gp:
#                             print("no")
#                      # print(model)
#                      group = f5['{}/score/'.format(model)]
#                      print(group)
#               f5.close()
#               f5 = h5py.File(hdf5, 'a')
#               print(f5)
#               f5.close()
# print(f['1ACB_cm-it0_9890/score'][()])
# f5 = h5py.File(hdf5, 'a')
# database=

# from tmp import GINet
# from NeuralNet import NeuralNet
# # database = './1ATN_residue.hdf5'
# database ='./1ACB_residue.hdf5'
# edge_feature=['dist']
# node_feature=['type', 'polarity', 'bsa', 'pssm']
# pos=['pos']
# target='fnat'
# task='reg' 
# batch_size=2
# shuffle=True
# lr=0.001
# # add_target(graph_path=database, target_name='fnat',
# #        target_list='./data/target/1ATN/dummy_target.csv')

# model = NeuralNet(database, GINet,
#                node_feature=node_feature,
#                edge_feature=edge_feature,
#                target=target,
#                index=None,
#                task=task, 
#                lr=lr,
#                batch_size=batch_size,
#                shuffle=shuffle,
#                pos=pos
#                )

# model.train(nepoch=1,  hdf5='output.hdf5')