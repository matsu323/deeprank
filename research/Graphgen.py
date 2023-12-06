from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.tools.CustomizeGraph import add_target
pdb_path = './data/pdb/1ATN/'
pssm_path = './data/pssm/1ATN/'
ref = './data/ref/1ATN/'
# GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path,
#             graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)
GraphHDF5(pdb_path=pdb_path,graph_type='residue', outfile='1ATN_residue.hdf5',
          nproc=4, tmpdir='./tmpdir')
add_target(graph_path='.', target_name='bin_class',
      target_list='./data/target/1ATN/dummy_target.csv')