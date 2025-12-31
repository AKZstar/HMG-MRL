import matplotlib.pyplot as plt
plt.switch_backend('agg')
import deepchem as dc

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from .Featurizer_atom_bond import *
import pickle
import time
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from rdkit.Chem.Draw import SimilarityMaps
from io import StringIO
from config import cfg
from other_utils import rbf_transform
from .motif_utils import map_molecule_to_all_fragment_types
from rdkit.Chem import BRICS
from collections import defaultdict
from rdkit import Chem
import rdkit.Chem.Recap as Recap
from collections import defaultdict
import time
import traceback
DETAILED_DEBUG_ENABLED = True
smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5, 6]
bond_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            if self.ntype=='bond' and neighbor.ntype=='bond':
                pass
            else:
                neighbor._neighbors.append(self)

    def get_neighbors(self, neighbor_ntype):
        if self.ntype=='bond' and neighbor_ntype=='bond':
            bond_neighbor = {}
            atom_keys = ['a1','a2']
            atom_index = 0
            atom_len = 0
            for ii in range(len(self._neighbors)):
                if ii==0:
                    assert self._neighbors[ii].ntype == 'atom'
                if self._neighbors[ii].ntype == 'atom':
                    atom_len += 1
            assert atom_len==2
            for item in self._neighbors:
                if item.ntype == 'atom':
                    bond_neighbor[atom_keys[atom_index]] = []
                    atom_index += 1
                elif item.ntype == 'bond':
                    bond_neighbor[atom_keys[atom_index - 1]].append(item)
                elif item.ntype == 'molecule_bond':
                    break

            return bond_neighbor
        else:
            return [n for n in self._neighbors if n.ntype == neighbor_ntype]

class MolGraph(object):
    def __init__(self,mol,conv_3d):
        self.nodes = {} 
        self.mol = mol
        self.conv_3d = conv_3d
        self.frags_atom_bond_indices_dict = {}
        self.frag_feature_array = None

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self):
        nodes_by_degree_atom = {i: [] for i in degrees}
        nodes_by_degree_bond = {i: [] for i in bond_degrees}

        for node in self.nodes['atom']:
            nodes_by_degree_atom[len(node.get_neighbors('atom'))].append(node)
        for node in self.nodes['bond']:
            nodes_by_degree_bond[len(node.get_neighbors('bond')['a1'])+len(node.get_neighbors('bond')['a2'])].append(node)

        new_atom_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree_atom[degree]
            self.nodes[('atom', degree)] = cur_nodes
            new_atom_nodes.extend(cur_nodes)
        self.nodes['atom'] = new_atom_nodes

        atom_id_2_rdkit = {index: element.rdkit_ix for index, element in enumerate(self.nodes['atom'])}

        new_bond_nodes = []
        for degree in bond_degrees:
            cur_nodes = nodes_by_degree_bond[degree] 
            self.nodes[('bond', degree)] = cur_nodes
            new_bond_nodes.extend(cur_nodes)
        self.nodes['bond'] = new_bond_nodes
        bond_id_2_rdkit = {index: element.rdkit_ix for index, element in enumerate(self.nodes['bond'])}

        return atom_id_2_rdkit, bond_id_2_rdkit

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self,ntype):
        return np.array([node.rdkit_ix for node in self.nodes[ntype]])

    def neighbor_list(self, self_ntype, neighbor_ntype):

        assert self_ntype in self.nodes and neighbor_ntype in self.nodes

        if self_ntype=='bond' and neighbor_ntype=='bond':
            bond_neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
            bond_neighbor_list = []
            for self_node in self.nodes[self_ntype]:
                new_neighbor_dict = {}
                neighbor_dict = self_node.get_neighbors(neighbor_ntype)
                new_neighbor_dict['a1'] = [bond_neighbor_idxs[neighbor] for neighbor in neighbor_dict['a1']]
                new_neighbor_dict['a2'] = [bond_neighbor_idxs[neighbor] for neighbor in neighbor_dict['a2']]
                bond_neighbor_list.append(new_neighbor_dict)

            return bond_neighbor_list
        else:
            neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
            return [[neighbor_idxs[neighbor]
                    for neighbor in self_node.get_neighbors(neighbor_ntype)]
                    for self_node in self.nodes[self_ntype]]

def create_fragment_features_with_dims(fragment_map,
                                       atoms_by_rd_idx, bonds_by_rd_idx,
                                       atom_feature_dim, bond_feature_dim):
    if not isinstance(fragment_map, dict) or \
            not isinstance(atoms_by_rd_idx, dict) or \
            not isinstance(bonds_by_rd_idx, dict):
        print("错误(create_frag_features)：输入参数 fragment_map, atoms_by_rd_idx, bonds_by_rd_idx 应为字典。")
        return np.array([])
    if "whole_mol" not in fragment_map:
        print("错误(create_frag_features)：fragment_map 中缺少必需的 'whole_mol' 键。")
        return np.array([])
    if not isinstance(atom_feature_dim, int) or atom_feature_dim < 0:
        print(f"错误(create_frag_features)：atom_feature_dim ({atom_feature_dim}) 必须是一个非负整数。")
        return np.array([])
    if not isinstance(bond_feature_dim, int) or bond_feature_dim < 0:
        print(f"错误(create_frag_features)：bond_feature_dim ({bond_feature_dim}) 必须是一个非负整数。")
        return np.array([])

    whole_mol_data = fragment_map["whole_mol"]
    original_mol_feature = calculate_feature_vector(
        whole_mol_data[0], whole_mol_data[1],
        atoms_by_rd_idx, bonds_by_rd_idx,
        atom_feature_dim, bond_feature_dim
    )

    final_feature_list = [original_mol_feature]
    num_fragments = len(fragment_map) - 1

    if num_fragments > 0:
        fragment_keys = sorted([k for k in fragment_map.keys() if k != "whole_mol"],
                               key=lambda x: int(x.split('_')[1]))
        for key in fragment_keys:
            atom_indices, bond_indices = fragment_map[key]
            fragment_feature = calculate_feature_vector(
                atom_indices, bond_indices,
                atoms_by_rd_idx, bonds_by_rd_idx,
                atom_feature_dim, bond_feature_dim
            )
            final_feature_list.append(fragment_feature)

    try:
        if not final_feature_list or all(
                isinstance(feat, np.ndarray) and feat.size == 0 for feat in final_feature_list):
            if atom_feature_dim + bond_feature_dim > 0 and num_fragments > -1:
                print(
                    f"警告(create_frag_features): 特征列表为空或所有特征均为空，但期望维度为 ({atom_feature_dim + bond_feature_dim})。")
            return np.array(final_feature_list,
                            dtype=object if any(isinstance(f, list) for f in final_feature_list) else np.float32)

        if atom_feature_dim + bond_feature_dim == 0:
            return np.array(final_feature_list, dtype=object)

        return np.array(final_feature_list, dtype=np.float32)
    except ValueError as ve:
        print(f"错误(create_frag_features): 转换特征列表为NumPy数组时出错: {ve}")
        return np.array(final_feature_list, dtype=object)


def calculate_feature_vector(atom_indices, bond_indices,
                             atoms_by_rd_idx, bonds_by_rd_idx,
                             atom_feature_dim, bond_feature_dim):
    if atom_feature_dim > 0:
        atom_feature_sum = np.zeros(atom_feature_dim, dtype=np.float32)
        if atom_indices:
            for atom_idx in atom_indices:
                try:
                    atom_node_features = atoms_by_rd_idx[atom_idx].features
                    if not isinstance(atom_node_features, np.ndarray):
                        atom_node_features = np.array(atom_node_features, dtype=np.float32)
                    if atom_node_features.shape == (atom_feature_dim,):
                        atom_feature_sum += atom_node_features
                except Exception:
                    pass
    else:
        atom_feature_sum = np.array([], dtype=np.float32)

    if bond_feature_dim > 0:
        bond_feature_vec = np.zeros(bond_feature_dim, dtype=np.float32)
        if bond_indices:
            for bond_idx in bond_indices:
                try:
                    bond_node_features = bonds_by_rd_idx[bond_idx].features
                    if not isinstance(bond_node_features, np.ndarray):
                        bond_node_features = np.array(bond_node_features, dtype=np.float32)
                    if bond_node_features.shape == (bond_feature_dim,):
                        bond_feature_vec += bond_node_features
                except Exception:
                    pass
    else:
        bond_feature_vec = np.array([], dtype=np.float32)

    if atom_feature_sum.size > 0 and bond_feature_vec.size > 0:
        return np.concatenate((atom_feature_sum, bond_feature_vec))
    elif atom_feature_sum.size > 0:
        return atom_feature_sum
    elif bond_feature_vec.size > 0:
        return bond_feature_vec
    else:
        if atom_feature_dim + bond_feature_dim > 0:
            return np.zeros(atom_feature_dim + bond_feature_dim, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)



def graph_from_smiles(mol, conf_3d):
    graph = MolGraph(mol, conf_3d)
    atoms_by_rd_idx = {}
    bonds_by_rd_idx = {}
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom_idx)
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        new_bond_node = graph.new_node('bond', features=bond_features(conf_3d, bond),rdkit_ix=bond_idx)
        bonds_by_rd_idx[bond.GetIdx()] = new_bond_node
    for bond in mol.GetBonds():
        bond_node = bonds_by_rd_idx[bond.GetIdx()]

        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        bond_node.add_neighbors((atom1_node, ))
        _a1 = [b for b in bond.GetBeginAtom().GetBonds() if b.GetIdx() != bond.GetIdx()]
        a1 = [bonds_by_rd_idx[b.GetIdx()] for b in _a1]
        bond_node.add_neighbors(a1)

        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        bond_node.add_neighbors((atom2_node,))
        _a2 = [b for b in bond.GetEndAtom().GetBonds() if b.GetIdx() != bond.GetIdx()]
        a2 = [bonds_by_rd_idx[b.GetIdx()] for b in _a2]
        bond_node.add_neighbors(a2)

        atom1_node.add_neighbors((atom2_node,))
    mol_node = graph.new_node('molecule_atom')
    mol_node.add_neighbors(graph.nodes['atom'])

    mol_edge = graph.new_node('molecule_bond') 
    mol_edge.add_neighbors(graph.nodes['bond'])
    smarts_rules_file = os.path.join(os.path.dirname(__file__), 'fg_dicts.txt')
    graph.frags_atom_bond_indices_dict = map_molecule_to_all_fragment_types(
        mol,
        smarts_rules_file
    )
    graph.frag_feature_array = create_fragment_features_with_dims(
        graph.frags_atom_bond_indices_dict,
        atoms_by_rd_idx,bonds_by_rd_idx,cfg.atom_dim_in,cfg.bond_dim_in
    )
    return graph
def array_rep_from_smiles(molgraph,atom_id_2_rdkit, bond_id_2_rdkit):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    degrees = [0,1,2,3,4,5,6]
    bond_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),

                'frag_feature_array' :molgraph.frag_feature_array,
                'frags_atom_bond_indices_dict':molgraph.frags_atom_bond_indices_dict,

                'atom_list'     : molgraph.neighbor_list('molecule_atom', 'atom'), 
                'bond_list': molgraph.neighbor_list('molecule_bond', 'bond'),
                'rdkit_ix_atom'      : molgraph.rdkit_ix_array('atom'),
                'rdkit_ix_bond': molgraph.rdkit_ix_array('bond'),
                'mol_obj' : molgraph.mol,
                'conv_3d' : molgraph.conv_3d,
                'atom_id_2_rdkit' : atom_id_2_rdkit,
                'bond_id_2_rdkit' : bond_id_2_rdkit}

    for degree in degrees:
        arrayrep[('atom_neighbors_atom', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('atom_neighbors_bond', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    arrayrep['bond_neighbor_bond'] = molgraph.neighbor_list('bond','bond')
    arrayrep['bond_neighbor_atom'] = np.array(molgraph.neighbor_list('bond', 'atom'), dtype=int)

    return arrayrep


def gen_descriptor_data(smilesList):
    smiles_to_fingerprint_array = {}
    not_sucess_smiles = []
    print("初始smileslist len： ",len(smilesList))
    sucess_smiles = []

    for i, smiles in enumerate(smilesList):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            mol = Chem.MolFromSmiles(smiles)
            new_mol = Chem.AddHs(mol)
            numConfs_ = AllChem.EmbedMultipleConfs(new_mol, numConfs=cfg.numConfs)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol, maxIters=cfg.MMFF_maxIters)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            conf_3d = new_mol.GetConformer(id=int(index))
            new_mol = Chem.MolFromSmiles(smiles)

            molgraph = graph_from_smiles(new_mol, conf_3d)
            atom_id_2_rdkit, bond_id_2_rdkit = molgraph.sort_nodes_by_degree()
            arrayrep = array_rep_from_smiles(molgraph, atom_id_2_rdkit, bond_id_2_rdkit)
            smiles_to_fingerprint_array[smiles] = arrayrep
            sucess_smiles.append(smiles)

        except Exception as e:
            print(smiles)
            print('wrong')
            not_sucess_smiles.append(smiles)
            time.sleep(3)
            continue


    print("成功完成smileslit 长度： ", len(sucess_smiles))
    print("smiles_to_fingerprint_array keys length: ", len(smiles_to_fingerprint_array.keys()))
    return smiles_to_fingerprint_array



def get_smiles_dicts(smilesList):
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_atom_list = {}
    smiles_to_rdkit_bond_list = {}
    smiles_to_frag_atom_bond_incices = {}
    max_frag_len = 80

    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features'] 
        bond_features = arrayrep['bond_features'] 

        rdkit_atom_list = arrayrep['rdkit_ix_atom'] 
        rdkit_bond_list = arrayrep['rdkit_ix_bond'] 
        smiles_to_rdkit_atom_list[smiles] = rdkit_atom_list
        smiles_to_rdkit_bond_list[smiles] = rdkit_bond_list

        smiles_to_frag_atom_bond_incices[smiles] = arrayrep['frags_atom_bond_indices_dict']

        atom_len, num_atom_features = atom_features.shape  
        bond_len, num_bond_features = bond_features.shape  

        if atom_len > max_atom_len:
            max_atom_len = atom_len 
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_frag_info = {}

    smiles_to_atom_neighbors_atom = {} 
    smiles_to_atom_neighbors_bond = {}  

    smiles_to_bond_neighbors_bond = {}  
    smiles_to_bond_neighbors_atom = {}  

    smiles_to_atom_mask = {}
    smiles_to_bond_mask = {}  

    smiles_to_frag_mask = {}
    smiles_to_descriptor = {}

    degrees = [0, 1, 2, 3, 4, 5, 6]
    bond_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    descriptor = dc.feat.RDKitDescriptors(is_normalized=True)

    max_edge_ids_len = 0

    # then run through our numpy array again
    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_mask = np.zeros((max_atom_len)) 
        bond_mask = np.zeros((max_bond_len)) 

        frag_mask = np.zeros((max_frag_len))

        atoms = np.zeros((max_atom_len,
                          num_atom_features)) 
        bonds = np.zeros((max_bond_len, num_bond_features))

        frag_features = np.zeros((max_frag_len, cfg.atom_dim_in+cfg.bond_dim_in))

        atom_neighbors_atom = np.zeros((max_atom_len, len(degrees)))
        atom_neighbors_bond = np.zeros((max_atom_len, len(degrees)))

        atom_neighbors_atom.fill(max_atom_index_num)  
        atom_neighbors_bond.fill(max_bond_index_num) 

        bond_neighbors_bond = np.zeros((max_bond_len, len(bond_degrees)))
        bond_neighbors_atom = np.zeros((max_bond_len, len(bond_degrees)))

        bond_neighbors_bond.fill(max_bond_index_num)
        bond_neighbors_atom.fill(max_atom_index_num) 

        atom_features = arrayrep['atom_features'] # 每个分子的化学键特征数组，二维

        frag_feature_array = arrayrep['frag_feature_array']

        for i, feature in enumerate(atom_features):
            atom_mask[i] = 1.0 
            atoms[i] = feature

        for j, feature in enumerate(bond_features):
            bond_mask[j] = 1.0 
            bonds[j] = feature 

        for i, feature in enumerate(frag_feature_array):
            print(smiles)
            frag_mask[i] = 1.0 
            frag_features[i] = feature

        atom_neighbor_atom_count = 0
        atom_neighbor_bond_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_atom_list = arrayrep[('atom_neighbors_atom', degree)]
            atom_neighbors_bond_list = arrayrep[('atom_neighbors_bond', degree)]

            if len(atom_neighbors_atom_list) > 0:

                for i, degree_array in enumerate(atom_neighbors_atom_list):
                    for j, value in enumerate(degree_array):
                        atom_neighbors_atom[atom_neighbor_atom_count, j] = value
                    atom_neighbor_atom_count += 1

            if len(atom_neighbors_bond_list) > 0:
                for i, degree_array in enumerate(atom_neighbors_bond_list):
                    for j, value in enumerate(degree_array):
                        atom_neighbors_bond[atom_neighbor_bond_count, j] = value
                    atom_neighbor_bond_count += 1

        bond_neighbor_bond_list = arrayrep['bond_neighbor_bond']
        bond_neighbor_atom_list = arrayrep['bond_neighbor_atom']
        if len(bond_neighbor_bond_list) > 0:
            for i, bond_neighbor_dict in enumerate(bond_neighbor_bond_list):
                a_1 = bond_neighbor_dict['a1']
                a_2 = bond_neighbor_dict['a2']

                for j, value in enumerate(a_1):
                    bond_neighbors_bond[i, j] = value
                for j, value in enumerate(a_2):
                    bond_neighbors_bond[i, j + 7] = value

        if len(bond_neighbor_atom_list) > 0:
            for i, bond_neighbor_atom_array in enumerate(bond_neighbor_atom_list):
                for j, value in enumerate(bond_neighbor_atom_array):
                    if j == 0:
                        bond_neighbors_atom[i, j] = value
                    elif j == 1:
                        bond_neighbors_atom[i, 7] = value

        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds

        smiles_to_frag_info[smiles] = frag_features

        smiles_to_atom_neighbors_atom[smiles] = atom_neighbors_atom
        smiles_to_atom_neighbors_bond[smiles] = atom_neighbors_bond

        smiles_to_bond_neighbors_bond[smiles] = bond_neighbors_bond
        smiles_to_bond_neighbors_atom[smiles] = bond_neighbors_atom

        smiles_to_atom_mask[smiles] = atom_mask
        smiles_to_bond_mask[smiles] = bond_mask

        smiles_to_frag_mask[smiles] = frag_mask

        smiles_to_descriptor[smiles] = descriptor.featurize(smiles)[0]

    del smiles_to_fingerprint_features
    feature_dicts = {}
    feature_dicts = {
        'smiles_to_frag_mask': smiles_to_frag_mask,
        'smiles_to_frag_info': smiles_to_frag_info,

        'smiles_to_frag_atom_bond_incices': smiles_to_frag_atom_bond_incices,

        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_bond_mask': smiles_to_bond_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors_atom': smiles_to_atom_neighbors_atom,
        'smiles_to_atom_neighbors_bond': smiles_to_atom_neighbors_bond,
        'smiles_to_bond_neighbors_bond': smiles_to_bond_neighbors_bond,
        'smiles_to_bond_neighbors_atom': smiles_to_bond_neighbors_atom,
        'smiles_to_rdkit_atom_list': smiles_to_rdkit_atom_list,
        'smiles_to_rdkit_bond_list': smiles_to_rdkit_bond_list,
        'smiles_to_descriptor': smiles_to_descriptor,
    }
    return feature_dicts


def save_smiles_dicts(smilesList,filename):
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0  
    num_bond_features = 0  
    smiles_to_rdkit_atom_list = {}
    smiles_to_rdkit_bond_list = {}

    smiles_to_frag_atom_bond_incices = {}

    max_frag_len = 80

    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features'] 
        bond_features = arrayrep['bond_features']  

        rdkit_atom_list = arrayrep['rdkit_ix_atom']  
        rdkit_bond_list = arrayrep['rdkit_ix_bond'] 
        smiles_to_rdkit_atom_list[smiles] = rdkit_atom_list
        smiles_to_rdkit_bond_list[smiles] = rdkit_bond_list

        smiles_to_frag_atom_bond_incices[smiles] = arrayrep['frags_atom_bond_indices_dict']

        atom_len, num_atom_features = atom_features.shape 
        bond_len, num_bond_features = bond_features.shape  

        if atom_len > max_atom_len:
            max_atom_len = atom_len  
        if bond_len > max_bond_len:
            max_bond_len = bond_len  

    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_frag_info = {}

    smiles_to_atom_neighbors_atom = {} 
    smiles_to_atom_neighbors_bond = {}  

    smiles_to_bond_neighbors_bond = {}  
    smiles_to_bond_neighbors_atom = {}  

    smiles_to_atom_mask = {}
    smiles_to_bond_mask = {}  

    smiles_to_frag_mask = {}
    smiles_to_descriptor = {}

    degrees = [0, 1, 2, 3, 4, 5, 6]
    bond_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    descriptor = dc.feat.RDKitDescriptors(is_normalized=True)

    max_edge_ids_len = 0

    # then run through our numpy array again
    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_mask = np.zeros((max_atom_len)) 
        bond_mask = np.zeros((max_bond_len))

        frag_mask = np.zeros((max_frag_len))
        atoms = np.zeros((max_atom_len,
                          num_atom_features))  
        bonds = np.zeros((max_bond_len, num_bond_features))

        frag_features = np.zeros((max_frag_len, cfg.atom_dim_in+cfg.bond_dim_in))

        atom_neighbors_atom = np.zeros((max_atom_len, len(degrees)))
        atom_neighbors_bond = np.zeros((max_atom_len, len(degrees)))

        atom_neighbors_atom.fill(max_atom_index_num)  
        atom_neighbors_bond.fill(max_bond_index_num)  
        bond_neighbors_bond = np.zeros((max_bond_len, len(bond_degrees)))
        bond_neighbors_atom = np.zeros((max_bond_len, len(bond_degrees)))

        bond_neighbors_bond.fill(max_bond_index_num) 
        bond_neighbors_atom.fill(max_atom_index_num)  

        atom_features = arrayrep['atom_features'] 
        bond_features = arrayrep['bond_features'] 

        frag_feature_array = arrayrep['frag_feature_array']

        for i, feature in enumerate(atom_features):
            atom_mask[i] = 1.0  
            atoms[i] = feature  

        for j, feature in enumerate(bond_features):
            bond_mask[j] = 1.0 
            bonds[j] = feature 

        for i, feature in enumerate(frag_feature_array):
            print(smiles)
            frag_mask[i] = 1.0  
            frag_features[i] = feature

        atom_neighbor_atom_count = 0
        atom_neighbor_bond_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_atom_list = arrayrep[('atom_neighbors_atom', degree)]
            atom_neighbors_bond_list = arrayrep[('atom_neighbors_bond', degree)]

            if len(atom_neighbors_atom_list) > 0:

                for i, degree_array in enumerate(atom_neighbors_atom_list):
                    for j, value in enumerate(degree_array):
                        atom_neighbors_atom[atom_neighbor_atom_count, j] = value
                    atom_neighbor_atom_count += 1

            if len(atom_neighbors_bond_list) > 0:
                for i, degree_array in enumerate(atom_neighbors_bond_list):
                    for j, value in enumerate(degree_array):
                        atom_neighbors_bond[atom_neighbor_bond_count, j] = value
                    atom_neighbor_bond_count += 1

        bond_neighbor_bond_list = arrayrep['bond_neighbor_bond']
        bond_neighbor_atom_list = arrayrep['bond_neighbor_atom']
        if len(bond_neighbor_bond_list) > 0:
            for i, bond_neighbor_dict in enumerate(bond_neighbor_bond_list):
                a_1 = bond_neighbor_dict['a1']
                a_2 = bond_neighbor_dict['a2']

                for j, value in enumerate(a_1):
                    bond_neighbors_bond[i, j] = value
                for j, value in enumerate(a_2):
                    bond_neighbors_bond[i, j + 7] = value

        if len(bond_neighbor_atom_list) > 0:
            for i, bond_neighbor_atom_array in enumerate(bond_neighbor_atom_list):
                for j, value in enumerate(bond_neighbor_atom_array):
                    if j == 0:
                        bond_neighbors_atom[i, j] = value
                    elif j == 1:
                        bond_neighbors_atom[i, 7] = value

        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds

        smiles_to_frag_info[smiles] = frag_features

        smiles_to_atom_neighbors_atom[smiles] = atom_neighbors_atom
        smiles_to_atom_neighbors_bond[smiles] = atom_neighbors_bond

        smiles_to_bond_neighbors_bond[smiles] = bond_neighbors_bond
        smiles_to_bond_neighbors_atom[smiles] = bond_neighbors_atom

        smiles_to_atom_mask[smiles] = atom_mask
        smiles_to_bond_mask[smiles] = bond_mask

        smiles_to_frag_mask[smiles] = frag_mask

        smiles_to_descriptor[smiles] = descriptor.featurize(smiles)[0]
    del smiles_to_fingerprint_features
    feature_dicts = {}
    feature_dicts = {
        'smiles_to_frag_mask': smiles_to_frag_mask,
        'smiles_to_frag_info': smiles_to_frag_info,

        'smiles_to_frag_atom_bond_incices': smiles_to_frag_atom_bond_incices,

        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_bond_mask': smiles_to_bond_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors_atom': smiles_to_atom_neighbors_atom,
        'smiles_to_atom_neighbors_bond': smiles_to_atom_neighbors_bond,
        'smiles_to_bond_neighbors_bond': smiles_to_bond_neighbors_bond,
        'smiles_to_bond_neighbors_atom': smiles_to_bond_neighbors_atom,
        'smiles_to_rdkit_atom_list': smiles_to_rdkit_atom_list,
        'smiles_to_rdkit_bond_list': smiles_to_rdkit_bond_list,
        'smiles_to_descriptor': smiles_to_descriptor,
    }

    pickle.dump(feature_dicts,open(filename+'.pickle',"wb"))
    print('feature dicts file saved as '+ filename+'.pickle')

    return feature_dicts


def get_smiles_array(smilesList, feature_dicts):
    atom_mask = []
    bond_mask = []

    frag_mask = []

    x_atoms = []
    x_bonds = []

    frag_features = []
    atom_neighbors_atom_index = []
    atom_neighbors_bond_index = []
    bond_neighbors_bond_index = []
    bond_neighbors_atom_index = []
    molecule_descriptor = []

    for smiles in smilesList:
        atom_mask.append(feature_dicts['smiles_to_atom_mask'][smiles])
        bond_mask.append(feature_dicts['smiles_to_bond_mask'][smiles])

        frag_mask.append(feature_dicts['smiles_to_frag_mask'][smiles])
        x_atoms.append(feature_dicts['smiles_to_atom_info'][smiles])
        x_bonds.append(feature_dicts['smiles_to_bond_info'][smiles])
        frag_features.append(feature_dicts['smiles_to_frag_info'][smiles])

        atom_neighbors_atom_index.append(feature_dicts['smiles_to_atom_neighbors_atom'][smiles])
        atom_neighbors_bond_index.append(feature_dicts['smiles_to_atom_neighbors_bond'][smiles])
        bond_neighbors_bond_index.append(feature_dicts['smiles_to_bond_neighbors_bond'][smiles])
        bond_neighbors_atom_index.append(feature_dicts['smiles_to_bond_neighbors_atom'][smiles])

        molecule_descriptor.append(feature_dicts['smiles_to_descriptor'][smiles])



    return np.asarray(x_atoms), np.asarray(x_bonds), np.asarray(frag_features), np.asarray(
        atom_neighbors_atom_index), \
        np.asarray(atom_neighbors_bond_index), np.asarray(bond_neighbors_bond_index), \
        np.asarray(bond_neighbors_atom_index), np.asarray(
        atom_mask), np.asarray(bond_mask), np.asarray(frag_mask),\
        feature_dicts['smiles_to_frag_atom_bond_incices'], np.asarray(molecule_descriptor), feature_dicts['smiles_to_rdkit_atom_list'], feature_dicts['smiles_to_rdkit_bond_list']











