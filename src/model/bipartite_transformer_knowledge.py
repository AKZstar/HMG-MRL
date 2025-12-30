"""
双向图Transformer知识头模型
主模型架构，整合原子-键GNN和片段Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from act_func import act_dict
from layer_utils import *


from model_components import (
    PreGNN, PostGNN_classifier, clones,
    Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward,
    GNN_mol_atom_aggregate, GNN_mol_bond_aggregate,
    GNN_mol_atom_update, GNN_mol_bond_update,
    Frag_atom_cross_attention, info_nce_loss,
    Layer_attention_by_mol
)


class GNN_atom_bond(nn.Module):
   
    def __init__(self, atom_dim_in, bond_dim_in, output_units_num, learn_eps=True):
        super().__init__()

        self.atom_feature_dim = cfg.atom_feature_dim
        self.bond_feature_dim = cfg.bond_feature_dim
        self.radius = cfg.radius

        self.atom_preGNN = PreGNN(atom_dim_in, self.atom_feature_dim, cfg.preGNN_num_layers)
        self.bond_preGNN = PreGNN(bond_dim_in, self.bond_feature_dim, cfg.preGNN_num_layers)

        self.atom_neighbor_transform = nn.ModuleList([
            nn.Linear(self.atom_feature_dim, self.atom_feature_dim) for r in range(self.radius)
        ])
        self.bond_neighbor_transform = nn.ModuleList([
            nn.Linear(self.bond_feature_dim, self.bond_feature_dim) for r in range(self.radius)
        ])
        
        self.atom_neighbor_transform_concat = nn.ModuleList([
            nn.Linear(self.atom_feature_dim*2, self.atom_feature_dim) for r in range(self.radius)
        ])
        self.bond_neighbor_transform_concat = nn.ModuleList([
            nn.Linear(self.bond_feature_dim*2, self.bond_feature_dim) for r in range(self.radius)
        ])

        self.atom_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.atom_feature_dim, self.atom_feature_dim),
                nn.ReLU(),
            ) for _ in range(self.radius)
        ])
        self.bond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bond_feature_dim, self.bond_feature_dim),
                nn.ReLU(),
            ) for _ in range(self.radius)
        ])

        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(self.radius))
        else:
            self.register_buffer('eps', torch.zeros(self.radius))

        self.get_mol_atom_context = GNN_mol_atom_aggregate(self.atom_feature_dim)
        self.get_mol_atom_update = GNN_mol_atom_update(cfg.atom_feature_dim)

        self.get_mol_bond_context = GNN_mol_bond_aggregate(self.bond_feature_dim)
        self.get_mol_bond_update = GNN_mol_bond_update(cfg.bond_feature_dim)
        
        self.node_gru = nn.GRU(cfg.atom_feature_dim, cfg.atom_feature_dim, batch_first=True, 
                               bidirectional=True)

        self.frag_input_dim = cfg.atom_dim_in + cfg.bond_dim_in
        self.frag_cross_atten = Frag_atom_cross_attention(self.atom_feature_dim)

        self.frag_input_projection = nn.Linear(self.frag_input_dim, self.atom_feature_dim)

        attn_module = MultiHeadedAttention(h=cfg.frg_nhead, d_model=self.atom_feature_dim, dropout=cfg.dropout)
        ff_module = PositionwiseFeedForward(d_model=self.atom_feature_dim, d_ff=self.atom_feature_dim, dropout=cfg.dropout)
        encoder_layer_no_norm = EncoderLayer(size=self.atom_feature_dim, self_attn=attn_module, 
                                             feed_forward=ff_module, dropout=cfg.dropout)
        self.encoder_no_norm = Encoder(layer=encoder_layer_no_norm, N=cfg.frag_num_encoder_layers)

        self.projector_atom = nn.Sequential(
            nn.Linear(cfg.atom_feature_dim, cfg.atom_feature_dim, bias=False),
            nn.ReLU()
        )
        self.projector_frag = nn.Sequential(
            nn.Linear(cfg.atom_feature_dim, cfg.atom_feature_dim, bias=False),
            nn.ReLU()
        )

        self.get_mol_atom_context_des = GNN_mol_atom_aggregate(self.atom_feature_dim)
        self.get_mol_atom_update_des = GNN_mol_atom_update(cfg.atom_feature_dim)

        self.descriptor_mlp = nn.ModuleList([
            GeneralLinearLayer(200, cfg.atom_feature_dim, has_act=True, 
                              has_dropout=False, need_layernorm=False)
        ] + [
            GeneralLinearLayer(cfg.atom_feature_dim, cfg.atom_feature_dim, has_act=True, 
                              has_dropout=True, need_layernorm=False) 
            for _ in range(cfg.des_num_layers-1)
        ])

        self.postGNN_classifier = PostGNN_classifier(cfg.atom_feature_dim, output_units_num, cfg.classifier_numlayers)
        self.layer_atten = Layer_attention_by_mol()
        self.layer_update_act = act_dict[cfg.GNN_update_act]

    def forward(self, x_atoms, x_bonds, frag_src_features, atom_neighbors_atom_index, atom_neighbors_bond_index,
                bond_neighbors_bond_index, bond_neighbors_atom_index, atom_mask, bond_mask, frag_src_valid_mask, 
                molecule_descriptor):
        
        global mol_feature_by_atom, mol_feature_by_bond
        global op_mol_atom, op_mol_motif, op_mol_descrip

        batch_size, mol_atom_num, len_atom_feat = x_atoms.size()
        batch_size, mol_bond_num, len_bond_feat = x_bonds.size()

        atom_feature = self.atom_preGNN(x_atoms)
        bond_feature = self.bond_preGNN(x_bonds)

        atom_attend_mask, atom_softmax_mask = get_atom_attend_and_softmax_mask(atom_neighbors_atom_index, mol_atom_num)
        bond_attend_mask, bond_softmax_mask = get_bond_attend_and_softmax_mask(bond_neighbors_bond_index, mol_bond_num)
        mol_atom_attend_mask, mol_atom_softmax_mask = get_mol_atom_attend_and_softmax_mask(atom_mask)
        mol_bond_attend_mask, mol_bond_softmax_mask = get_mol_bond_attend_and_softmax_mask(bond_mask)

        for i in range(self.radius):
            atom_feature_nb = self.atom_neighbor_transform[i](atom_feature)
            bond_feature_nb = self.bond_neighbor_transform[i](bond_feature)

            atom_neighbor_feature = get_atom_neighbor_feature_atom_bond_new(
                atom_feature_nb, bond_feature_nb, atom_neighbors_atom_index, atom_neighbors_bond_index)
            bond_neighbor_feature = get_bond_neighbor_feature_bond_atom_new(
                atom_feature_nb, bond_feature_nb, bond_neighbors_bond_index, bond_neighbors_atom_index)

            atom_context = torch.sum(torch.mul(atom_attend_mask, atom_neighbor_feature), -1)
            atom_feature_ = self.atom_mlps[i]((1 + self.eps[i]) * atom_feature + atom_context)
            atom_feature = atom_feature_ + atom_feature

            bond_context = torch.sum(torch.mul(bond_attend_mask, bond_neighbor_feature), -2)
            bond_feature_ = self.bond_mlps[i]((1 + self.eps[i]) * bond_feature + bond_context)
            bond_feature = bond_feature_ + bond_feature
            
            if i == 0:
                op_mol_atom = torch.sum(atom_feature * mol_atom_attend_mask, dim=-2)

        frag_features = F.relu(self.frag_input_projection(frag_src_features))
        
        atom_hidden = atom_feature.max(1)[0].unsqueeze(0).repeat(2, 1, 1)
        atom_feature, atom_hidden = self.node_gru(atom_feature, atom_hidden)
        
        atom_feature_reshaped = atom_feature.view(batch_size, mol_atom_num, 2, cfg.atom_feature_dim)
        atom_feature = atom_feature_reshaped[:, :, 0, :] + atom_feature_reshaped[:, :, 1, :]

        frag_context = self.frag_cross_atten(frag_features, atom_feature, mol_atom_attend_mask, mol_atom_softmax_mask)
        frag_features = frag_features + frag_context

        frag_src_mask = frag_src_valid_mask
        frag_output, last_layer_attn_weights, op_motif = self.encoder_no_norm(frag_features, frag_src_mask)
        frag_cls = frag_output[:, 0, :]
        op_mol_motif = op_motif[:, 0, :]

        mol_feature_by_atom = torch.sum(atom_feature * mol_atom_attend_mask, dim=-2)
        mol_context_by_atom = self.get_mol_atom_context(mol_feature_by_atom, atom_feature, 
                                                         mol_atom_attend_mask, mol_atom_softmax_mask)
        mol_feature_by_atom = self.get_mol_atom_update(mol_feature_by_atom, mol_context_by_atom)

        mol_feature_by_atom_projH = self.projector_atom(mol_feature_by_atom)
        frag_cls_projH = self.projector_frag(frag_cls)

        mol_feature_by_atom_projH_norm = F.normalize(mol_feature_by_atom_projH, dim=-1)
        frag_cls_projH_norm = F.normalize(frag_cls_projH, dim=-1)

        ct_loss = None
        if cfg.ctloss_name == "infonce":
            features = torch.cat((mol_feature_by_atom_projH, frag_cls_projH), dim=0)
            ct_loss = info_nce_loss(features, batch_size)
        else:
            ct_loss = F.mse_loss(frag_cls_projH_norm, mol_feature_by_atom_projH_norm)

        for i in range(cfg.des_num_layers):
            molecule_descriptor = self.descriptor_mlp[i](molecule_descriptor)
            if i == 0:
                op_mol_descrip = molecule_descriptor

        mol_context_by_des = self.get_mol_atom_context_des(molecule_descriptor, atom_feature, 
                                                            mol_atom_attend_mask, mol_atom_softmax_mask)
        mol_feature_by_des = self.get_mol_atom_update_des(molecule_descriptor, mol_context_by_des)

        layer_neighbor_feature = torch.stack([frag_cls, mol_feature_by_atom, mol_feature_by_des], dim=1)
        mol_layer_feature_by_atom_sum = torch.sum(layer_neighbor_feature, dim=-2)
        mol_layer_context = self.layer_atten(mol_layer_feature_by_atom_sum, layer_neighbor_feature)
        mol_final_fea = self.layer_update_act(mol_layer_context + mol_layer_feature_by_atom_sum)

        output = self.postGNN_classifier(mol_final_fea)

        return atom_feature, output, ct_loss, op_mol_atom, op_mol_motif, op_mol_descrip, mol_final_fea