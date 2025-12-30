import matplotlib.pyplot as plt
plt.switch_backend('agg')
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import time
from config import cfg

from graph_utils import MolGraph, graph_from_smiles, array_rep_from_smiles
from fragment_utils import map_molecule_to_all_fragment_types

DETAILED_DEBUG_ENABLED = True
smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5, 6]
bond_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def gen_descriptor_data(smilesList):
    """生成描述符数据"""
    smiles_to_fingerprint_array = {}
    not_sucess_smiles = []

    print("初始smileslist len： ", len(smilesList))
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

            graph = MolGraph(new_mol, conf_3d)
            molgraph = graph_from_smiles(graph, new_mol, conf_3d)
            atom_id_2_rdkit, bond_id_2_rdkit = molgraph.sort_nodes_by_degree()
            arrayrep = array_rep_from_smiles(molgraph, atom_id_2_rdkit, bond_id_2_rdkit)
            smiles_to_fingerprint_array[smiles] = arrayrep

            sucess_smiles.append(smiles)

        except Exception as e:
            print(smiles)
            print('sheqi')
            not_sucess_smiles.append(smiles)
            time.sleep(3)
            continue

    print("成功完成smileslit 长度： ", len(sucess_smiles))
    print("smiles_to_fingerprint_array keys length: ", len(smiles_to_fingerprint_array.keys()))
    return smiles_to_fingerprint_array


def get_smiles_dicts(smilesList):
    """获取SMILES特征字典"""
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

    descriptor = dc.feat.RDKitDescriptors(is_normalized=True)

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_mask = np.zeros((max_atom_len))
        bond_mask = np.zeros((max_bond_len))
        frag_mask = np.zeros((max_frag_len))

        atoms = np.zeros((max_atom_len, num_atom_features))
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


def save_smiles_dicts(smilesList, filename):
    """保存SMILES特征字典到文件"""
    feature_dicts = get_smiles_dicts(smilesList) 
    pickle.dump(feature_dicts, open(filename+'.pickle', "wb"))
    print('feature dicts file saved as '+ filename+'.pickle')
    return feature_dicts


def get_smiles_array(smilesList, feature_dicts):
    """从特征字典获取SMILES数组"""
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
        feature_dicts['smiles_to_frag_atom_bond_incices'], np.asarray(molecule_descriptor), \
        feature_dicts['smiles_to_rdkit_atom_list'], feature_dicts['smiles_to_rdkit_bond_list']