pdb_path = './data_in/1ACB/'
pssm_path = './pssm/pssm_mapped_to_pdb/'
# ref = './data_in/1ATN/'
from GraphGenMP import GraphHDF5
from tools.CustomizeGraph import add_target
from NeuralNet import NeuralNet
from tmp import GINet
from time import time
t0=time()
# GraphHDF5(pdb_path=pdb_path,  pssm_path=pssm_path,biopython=False,
#           graph_type='residue', outfile='1ACB_residue.hdf5',
#           nproc=16, tmpdir='./tmpdir')

# add_target(graph_path='./1ACB_residue.hdf5', target_name='fnat',
#        target_list='./data_in/1ACB/Fnat.dat')

database = './1ACB_residue.hdf5'
t = time() - t0
print(t)
edge_feature=['dist']
node_feature=['type', 'polarity', 'bsa','ic',"charge",
              'pssm']
# node_feature=['type', 'polarity', 'bsa']
pos=['pos']
target='fnat'
task='reg' 
batch_size=100
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

model.train(nepoch=1,  hdf5='output.hdf5')

# from DataSet import HDF5DataSet
# print(HDF5DataSet())
