from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from config import cfg
from other_utils import rbf_transform
import os

import pickle



electronegativity_path = os.path.join(os.path.dirname(__file__), 'Electronegativity.pkl')
with open(electronegativity_path, 'rb') as f:
    edict = pickle.load(f)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


Atom_pair = [('C', 'C'),
             ('C', 'N'),
             ('C', 'O'),
             ('C', 'S'),
             ('N', 'N'),
             ('C', 'Cl'),
             ('O', 'S'),
             ('N', 'S'),
             ('N', 'O'),
             ('C', 'F'),
             ('Br', 'C'),
             ('O', 'P'),
             ('C', 'P'),
             ('S', 'S'),
             ('C', 'I'),
             ('C', 'Si'),
             ('N', 'P'),
             ('O', 'Si'),
             ('P', 'S'),
             ('C', 'Sn'),
             ('C', 'Se'),
             ('Cl', 'O'),
             ('O', 'O'),
             'other'
]

bond_degree = [0,1,2,3,4,5,6,7,8,'other']

def get_atom_pair(bond):
    atom1 = bond.GetBeginAtom().GetSymbol()
    atom2 = bond.GetEndAtom().GetSymbol()

    # Order does not matter, so we sort the pair
    atom_pair = tuple(sorted([atom1, atom2]))
    return atom_pair

def get_bond_degree(bond):
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    degree1 = atom1.GetDegree() - 1
    degree2 = atom2.GetDegree() - 1

    bond_degree = degree1 + degree2

    return bond_degree




def Electronegativity_diff(bond):
    atom1 = bond.GetBeginAtom().GetSymbol()
    atom2 = bond.GetEndAtom().GetSymbol()

    if atom1 != 'nodata' and atom2 != 'nodata':
        ediff = abs(edict[atom1] - edict[atom2])
    else:
        ediff = 10

    ediff = rbf_transform(ediff,cfg.ediff_center,cfg.ediff_gamma)

    return ediff


def get_bond_length(molconf, bondx):
    atom1 = bondx.GetBeginAtom()
    atom2 = bondx.GetEndAtom()
    bond_len = rdkit.Chem.rdMolTransforms.GetBondLength(molconf, atom1.GetIdx(), atom2.GetIdx())
    file_name = 'bond_length.txt'

    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write(str(bond_len))
    else:
        with open(file_name, 'a') as f:
            f.write(f',{bond_len}')
    bond_len = rbf_transform(bond_len, cfg.bond_len_centers, cfg.bond_len_gamma)

    return bond_len


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)

def bond_features(molconf, bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(get_atom_pair(bond), Atom_pair)
    bond_feats += one_of_k_encoding_unk(get_bond_degree(bond), bond_degree)
    bond_feats += list(Electronegativity_diff(bond))
    bond_feats += list(get_bond_length(molconf, bond))
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def num_atom_features():
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    a_feat = atom_features(a)
    return [len(atom_features(a)), a_feat]


def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return [len(bond_features(simple_mol.GetBonds()[0])),bond_features(simple_mol.GetBonds()[0])]


if __name__ == "__main__":
    print("atom_fea_len: ",num_atom_features())
    print("bond_fea_len: ",num_bond_features())
