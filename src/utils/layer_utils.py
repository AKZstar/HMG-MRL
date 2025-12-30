import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from act_func import act_dict


class GeneralLinearLayer(nn.Module):
    '''General wrapper for layers'''
    def __init__(self,
                 dim_in,
                 dim_out,
                 has_act=True,
                 has_dropout=True,
                 need_layernorm = False,
                 **kwargs):
        super().__init__()
        self.layer = nn.Linear(dim_in,dim_out)
        ####增加layer_normalization
        self.layer_norm = nn.LayerNorm(dim_out)
        self.need_layernorm = need_layernorm

        layer_wrapper = []
        if has_dropout and cfg.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=cfg.dropout))
        if has_act:
            layer_wrapper.append(act_dict[cfg.preGNN_act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if self.need_layernorm:
            batch = self.layer_norm(batch)
        batch = self.post_layer(batch)
        return batch


class GeneralMultiLinearLayer(nn.Module):
    '''General wrapper for stack of layers'''
    def __init__(self,
                 num_layers,
                 dim_in,
                 dim_out,
                 dim_inner=None,
                 final_act=False,
                 need_layernorm=False,
                 **kwargs):
        super().__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            # 对于第一层和最后一层，has_dropout设置为False
            has_dropout = False if (i == 0 or i == num_layers - 1)  else True
            layer = GeneralLinearLayer(d_in, d_out, has_act,has_dropout,need_layernorm,**kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


def get_atom_neighbor_feature_atom_bond(x_atoms, x_bonds, atom_neighbors_atom_index, atom_neighbors_bond_index):
    batch_size, mol_atom_num, len_atom_feat = x_atoms.size()

    atom_neighbors_bond = [x_bonds[i][atom_neighbors_bond_index[i]] for i in range(batch_size)]
    atom_neighbors_bond = torch.stack(atom_neighbors_bond, dim=0)
    atom_neighbors_atom = [x_atoms[i][atom_neighbors_atom_index[i]] for i in range(batch_size)]
    atom_neighbors_atom = torch.stack(atom_neighbors_atom, dim=0)

    atom_neighbor_feature = torch.cat([atom_neighbors_atom, atom_neighbors_bond], dim=-1)  # 将邻居原子特征与邻居化学键特征进行concat

    return atom_neighbor_feature


def get_atom_neighbor_feature_atom_bond_new(x_atoms, x_bonds, atom_neighbors_atom_index, atom_neighbors_bond_index):
    batch_size, mol_atom_num, len_atom_feat = x_atoms.size()

    atom_neighbors_bond = [x_bonds[i][atom_neighbors_bond_index[i]] for i in range(batch_size)]
    atom_neighbors_bond = torch.stack(atom_neighbors_bond, dim=0)
    atom_neighbors_atom = [x_atoms[i][atom_neighbors_atom_index[i]] for i in range(batch_size)]
    atom_neighbors_atom = torch.stack(atom_neighbors_atom, dim=0)

    # atom_neighbor_feature = torch.cat([atom_neighbors_atom, atom_neighbors_bond], dim=-1)  # 将邻居原子特征与邻居化学键特征进行concat
    atom_neighbor_feature = atom_neighbors_atom + atom_neighbors_bond
    
    return atom_neighbor_feature


def get_atom_neighbor_feature_atom(x_atoms, atom_neighbors_atom_index):
    batch_size, mol_atom_num, len_atom_feat = x_atoms.size()

    atom_neighbors_atom = [x_atoms[i][atom_neighbors_atom_index[i]] for i in range(batch_size)]
    atom_neighbors_atom = torch.stack(atom_neighbors_atom, dim=0)

    return atom_neighbors_atom


def get_node_neighbor_feature_node(x_atoms, atom_neighbors_atom_index):
    batch_size, mol_atom_num, len_atom_feat = x_atoms.size()

    atom_neighbors_atom = [x_atoms[i][atom_neighbors_atom_index[i]] for i in range(batch_size)]
    atom_neighbors_atom = torch.stack(atom_neighbors_atom, dim=0)

    return atom_neighbors_atom


def get_atom_attend_and_softmax_mask(atom_neighbors_atom_index , mol_atom_num):
    atom_attend_mask = atom_neighbors_atom_index.clone()
    atom_attend_mask[atom_attend_mask != mol_atom_num - 1] = 1
    atom_attend_mask[atom_attend_mask == mol_atom_num - 1] = 0
    atom_attend_mask = atom_attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

    atom_softmax_mask = atom_neighbors_atom_index.clone()
    atom_softmax_mask[atom_softmax_mask != mol_atom_num - 1] = 0
    atom_softmax_mask[atom_softmax_mask == mol_atom_num - 1] = -9e8  # make the softmax value extremly small
    atom_softmax_mask = atom_softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

    return atom_attend_mask, atom_softmax_mask


def get_node_attend_and_softmax_mask(atom_neighbors_atom_index , mol_atom_num):
    atom_attend_mask = atom_neighbors_atom_index.clone()
    atom_attend_mask[atom_attend_mask != mol_atom_num - 1] = 1
    atom_attend_mask[atom_attend_mask == mol_atom_num - 1] = 0
    atom_attend_mask = atom_attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

    atom_softmax_mask = atom_neighbors_atom_index.clone()
    atom_softmax_mask[atom_softmax_mask != mol_atom_num - 1] = 0
    atom_softmax_mask[atom_softmax_mask == mol_atom_num - 1] = -9e8  # make the softmax value extremly small
    atom_softmax_mask = atom_softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

    return atom_attend_mask, atom_softmax_mask



def get_mol_node_attend_and_softmax_mask(atom_mask):
    atom_mask = atom_mask.unsqueeze(2)

    mol_softmax_mask_by_atom = atom_mask.clone()
    mol_softmax_mask_by_atom[mol_softmax_mask_by_atom == 0] = -9e8
    mol_softmax_mask_by_atom[mol_softmax_mask_by_atom == 1] = 0
    mol_softmax_mask_by_atom = mol_softmax_mask_by_atom.type(torch.cuda.FloatTensor)

    return atom_mask, mol_softmax_mask_by_atom


def get_mol_atom_attend_and_softmax_mask(atom_mask):
    atom_mask = atom_mask.unsqueeze(2)

    mol_softmax_mask_by_atom = atom_mask.clone()
    mol_softmax_mask_by_atom[mol_softmax_mask_by_atom == 0] = -9e8
    mol_softmax_mask_by_atom[mol_softmax_mask_by_atom == 1] = 0
    mol_softmax_mask_by_atom = mol_softmax_mask_by_atom.type(torch.cuda.FloatTensor)

    return atom_mask, mol_softmax_mask_by_atom


def get_mol_bond_attend_and_softmax_mask(bond_mask):
    bond_mask = bond_mask.unsqueeze(2)

    mol_softmax_mask_by_bond = bond_mask.clone()
    mol_softmax_mask_by_bond[mol_softmax_mask_by_bond == 0] = -9e8
    mol_softmax_mask_by_bond[mol_softmax_mask_by_bond == 1] = 0
    mol_softmax_mask_by_bond = mol_softmax_mask_by_bond.type(torch.cuda.FloatTensor)

    return bond_mask, mol_softmax_mask_by_bond

def get_bond_neighbor_feature_bond_atom(x_atoms, x_bonds, bond_neighbors_bond_index, bond_neighbors_atom_index):
    batch_size, mol_bond_num, len_bond_feat = x_bonds.size()

    bond_neighbors_bond = [x_bonds[i][bond_neighbors_bond_index[i]] for i in range(batch_size)]
    bond_neighbors_bond = torch.stack(bond_neighbors_bond, dim=0)

    bond_neighbors_atom_index_fill = bond_neighbors_atom_index.clone()
    bond_neighbors_atom_index_fill[:, :, 1:7] = bond_neighbors_atom_index_fill[:, :, 0:1]##前6位的后5位都为与第一位相同，即化学键的第一个邻居原子特征
    bond_neighbors_atom_index_fill[:, :, 8:13] = bond_neighbors_atom_index_fill[:, :, 7:8]##后6位的后5位都与后6位的第一位相同，即化学键的第二个原子特征
    bond_neighbors_atom = [x_atoms[i][bond_neighbors_atom_index_fill[i]] for i in range(batch_size)]
    bond_neighbors_atom = torch.stack(bond_neighbors_atom, dim=0)

    bond_neighbor_feature = torch.cat([bond_neighbors_atom, bond_neighbors_bond], dim=-1)  # 将邻居原子特征与邻居化学键特征进行concat

    bond_attend_mask, _ = get_bond_attend_and_softmax_mask(bond_neighbors_bond_index , mol_bond_num)
    ######################################################真的需要吗#################################################################
    bond_neighbor_feature = bond_neighbor_feature * bond_attend_mask ######################################################真的需要吗

    return bond_neighbor_feature

def get_bond_neighbor_feature_bond_atom_new(x_atoms, x_bonds, bond_neighbors_bond_index, bond_neighbors_atom_index):
    batch_size, mol_bond_num, len_bond_feat = x_bonds.size()

    bond_neighbors_bond = [x_bonds[i][bond_neighbors_bond_index[i]] for i in range(batch_size)]
    bond_neighbors_bond = torch.stack(bond_neighbors_bond, dim=0)

    bond_neighbors_atom_index_fill = bond_neighbors_atom_index.clone()
    bond_neighbors_atom_index_fill[:, :, 1:7] = bond_neighbors_atom_index_fill[:, :, 0:1]##前6位的后5位都为与第一位相同，即化学键的第一个邻居原子特征
    bond_neighbors_atom_index_fill[:, :, 8:13] = bond_neighbors_atom_index_fill[:, :, 7:8]##后6位的后5位都与后6位的第一位相同，即化学键的第二个原子特征
    bond_neighbors_atom = [x_atoms[i][bond_neighbors_atom_index_fill[i]] for i in range(batch_size)]
    bond_neighbors_atom = torch.stack(bond_neighbors_atom, dim=0)

    bond_neighbor_feature = bond_neighbors_atom+bond_neighbors_bond  # 将邻居原子特征与邻居化学键特征进行concat

    bond_attend_mask, _ = get_bond_attend_and_softmax_mask(bond_neighbors_bond_index , mol_bond_num)
    ######################################################真的需要吗#################################################################
    bond_neighbor_feature = bond_neighbor_feature * bond_attend_mask ######################################################真的需要吗

    return bond_neighbor_feature


def get_bond_attend_and_softmax_mask(bond_neighbors_bond_index , mol_bond_num):
    bond_attend_mask = bond_neighbors_bond_index.clone()
    bond_attend_mask[bond_attend_mask != mol_bond_num - 1] = 1
    bond_attend_mask[bond_attend_mask == mol_bond_num - 1] = 0
    bond_attend_mask[:, :, [0, 7]] = 1
    bond_attend_mask = bond_attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

    bond_softmax_mask = bond_neighbors_bond_index.clone()
    bond_softmax_mask[bond_softmax_mask != mol_bond_num - 1] = 0
    bond_softmax_mask[bond_softmax_mask == mol_bond_num - 1] = -9e8
    bond_softmax_mask[:, :, [0, 7]] = 0
    bond_softmax_mask = bond_softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

    return bond_attend_mask, bond_softmax_mask

def get_bond_neighbor_feature_bond( x_bonds, bond_neighbors_bond_index):
    batch_size, mol_bond_num, len_bond_feat = x_bonds.size()

    bond_neighbors_bond = [x_bonds[i][bond_neighbors_bond_index[i]] for i in range(batch_size)]
    bond_neighbors_bond = torch.stack(bond_neighbors_bond, dim=0)

    bond_attend_mask, _ = get_bond_attend_and_softmax_mask(bond_neighbors_bond_index , mol_bond_num)
    ######################################################真的需要吗#################################################################
    bond_neighbor_feature = bond_neighbors_bond * bond_attend_mask ######################################################真的需要吗

    return bond_neighbor_feature


class GatedBimodalNN(nn.Module):
    def __init__(self, input_dim_v, input_dim_t, hidden_dim):
        super(GatedBimodalNN, self).__init__()
        # Define the weights for the visual modality
        self.Wv = nn.Linear(input_dim_v, hidden_dim)
        # Define the weights for the textual modality
        self.Wt = nn.Linear(input_dim_t, hidden_dim)
        # Define the weights for the gating mechanism
        self.Wz = nn.Linear(input_dim_v + input_dim_t, hidden_dim)

    def forward(self, xv, xt):
        # Compute the hidden states for each modality
        hv = torch.tanh(self.Wv(xv))
        ht = torch.tanh(self.Wt(xt))

        # Concatenate the inputs for the gating mechanism
        concatenated_inputs = torch.cat((xv, xt), dim=1)

        # Compute the gating signal
        z = torch.sigmoid(self.Wz(concatenated_inputs))

        # Compute the final output
        h = z * hv + (1 - z) * ht

        return h, z

    
class GatedBimodalNN_new_1(nn.Module):
    def __init__(self, input_dim_v, input_dim_t, hidden_dim):
        super(GatedBimodalNN_new_1, self).__init__()
        # Define the weights for the visual modality
        self.Wv = nn.Linear(input_dim_v, hidden_dim)
        # Define the weights for the textual modality
        self.Wt = nn.Linear(input_dim_t, hidden_dim)
        # Define the weights for the gating mechanism
        # self.Wz = nn.Linear(input_dim_v + input_dim_t, hidden_dim)

        self.fc_layer = nn.Sequential(
            nn.Linear(input_dim_v + input_dim_t, (input_dim_v + input_dim_t) // 6),
            nn.ReLU(inplace=True),
            nn.Linear((input_dim_v + input_dim_t) // 6, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, xv, xt):
        # Compute the hidden states for each modality
        hv = torch.tanh(self.Wv(xv))
        ht = torch.tanh(self.Wt(xt))

        # Concatenate the inputs for the gating mechanism
        concatenated_inputs = torch.cat((xv, xt), dim=1)

        # Compute the gating signal
        z = self.fc_layer(concatenated_inputs)

        # Compute the final output
        h = z * hv + (1 - z) * ht
        h = h + xv + xt

        return h, z


class GatedBimodalNN_new_2(nn.Module):
    def __init__(self, input_dim_v, input_dim_t, hidden_dim):
        super(GatedBimodalNN_new_2, self).__init__()
        # Define the weights for the visual modality
        self.Wv = nn.Linear(input_dim_v, hidden_dim)
        # Define the weights for the textual modality
        self.Wt = nn.Linear(input_dim_t, hidden_dim)
        # Define the weights for the gating mechanism
        self.Wz = nn.Linear(input_dim_v + input_dim_t, hidden_dim)

    def forward(self, xv, xt):
        # Compute the hidden states for each modality
        hv = torch.tanh(self.Wv(xv))
        ht = torch.tanh(self.Wt(xt))

        # Concatenate the inputs for the gating mechanism
        concatenated_inputs = torch.cat((xv, xt), dim=1)

        # Compute the gating signal
        z = torch.sigmoid(self.Wz(concatenated_inputs))

        # Compute the final output
        h = z * hv + (1 - z) * ht
        h = h + xv + xt

        return h, z
