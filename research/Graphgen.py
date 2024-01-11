from GraphGenMP import GraphHDF5
# from deeprank_gnn.GraphGenMP import GraphHDF5
from tools.CustomizeGraph import add_target
pdb_path = './exsample'
pssm_path = './pssm/pssm_mapped_to_pdb/'
# ref = './data/ref/1ATN/'
# GraphHDF5(pdb_path=pdb_path, ref_path=ref, pssm_path=pssm_path,
#             graph_type='residue', outfile='1ATN_residue.hdf5', nproc=4)

# a=GraphHDF5(pdb_path=pdb_path,pssm_path=pssm_path,biopython=False, graph_type='residue', outfile='1ACB_residue.hdf5',
#           nproc=4, tmpdir='./tmpdir')

# add_target(graph_path='.', target_name='bin_class',target_list='./data/target/1ATN/dummy_target.csv')