import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import cfg
from model.act_func import act_dict
from model.layer_utils import *
import os
from os.path import exists
from torch.nn.functional import log_softmax, pad
import copy


def PreGNN(dim_in, dim_out, num_layers):
    return GeneralMultiLinearLayer(num_layers,
                                   dim_in,
                                   dim_out,
                                   dim_inner=dim_out,
                                   final_act=True,
                                   need_layernorm=False)

def PostGNN_classifier(dim_in, dim_out, num_layers):
    if dim_in > 100:
        postGNN_featureLen = 64
    else:
        postGNN_featureLen = 32

    if cfg.classifier_reduction == True:
        postGNN_featureLen = postGNN_featureLen
    else:
        postGNN_featureLen = dim_in

    return GeneralMultiLinearLayer(num_layers,
                                   dim_in,
                                   dim_out,
                                   dim_inner=dim_in,
                                   final_act=False,
                                   need_layernorm=False)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        op_mol_motif=None
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, mask)
            
            if i==0:
                op_mol_motif=x
            
            if i == len(self.layers) - 1:
                last_layer_attn_weights = attn_weights
        return x, last_layer_attn_weights,op_mol_motif

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(x))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.sublayer[0](x, lambda x_ignored: attn_output)

        return self.sublayer[1](x, self.feed_forward), attn_weights

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_1(x).relu())

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return F.relu(self.linears[-1](x)), self.attn

class GNN_mol_atom_aggregate(nn.Module):
    def __init__(self, atom_feature_dim, need_Linear_trans=True):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.need_Linear_trans = need_Linear_trans
        self.attn_type = cfg.attn_type
        self.dropout = nn.Dropout(cfg.dropout)
        if need_Linear_trans:
            self.mol_atom_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
            self.mol_atom_neighbor_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        if self.attn_type == 'GAT':
            self.mol_GAT_align = nn.Linear(2 * self.atom_feature_dim, 1)
        elif self.attn_type == 'GATV2':
            self.mol_GATV2_align = nn.Linear(self.atom_feature_dim, 1)
        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]
        self.need_layer_norm = cfg.layer_norm
        self.layer_norm = nn.LayerNorm(self.atom_feature_dim)

    def forward(self, mol_atom_feature, atom_feature, mol_atom_attend_mask, mol_atom_softmax_mask):
        if self.need_Linear_trans == True:
            mol_atom_feature = self.dropout(mol_atom_feature)
            atom_feature = self.dropout(atom_feature)
            mol_atom_feature = self.mol_atom_fc(mol_atom_feature)
            atom_feature = self.mol_atom_neighbor_fc(atom_feature)

        batch_size, mol_atom_length, atom_feature_dim = atom_feature.shape
        mol_atom_expand = mol_atom_feature.unsqueeze(-2).expand(batch_size, mol_atom_length, atom_feature_dim)

        if self.attn_type == 'GAT':
            mol_feature_align = torch.cat([mol_atom_expand, atom_feature], dim=-1)
            mol_align_score = self.GNN_attn_act(self.mol_GAT_align(mol_feature_align))
            mol_align_score = mol_align_score + mol_atom_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * mol_atom_attend_mask
            mol_atom_context = torch.sum(torch.mul(mol_attention_weight, atom_feature), -2)
        elif self.attn_type == 'GATV2':
            mol_feature_align = mol_atom_expand + atom_feature
            mol_feature_align = self.GNN_attn_act(mol_feature_align)
            mol_align_score = self.mol_GATV2_align(mol_feature_align)
            mol_align_score = mol_align_score + mol_atom_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * mol_atom_attend_mask
            mol_atom_context = torch.sum(torch.mul(mol_attention_weight, atom_feature), -2)
        if self.need_layer_norm == True:
            mol_atom_context = self.layer_norm(mol_atom_context)

        return mol_atom_context

class GNN_mol_bond_aggregate(nn.Module):
    def __init__(self, bond_feature_dim, need_Linear_trans=True):
        super().__init__()
        self.bond_feature_dim = bond_feature_dim
        self.need_Linear_trans = need_Linear_trans
        self.attn_type = cfg.attn_type
        self.dropout = nn.Dropout(cfg.dropout)
        if need_Linear_trans:
            self.mol_bond_fc = nn.Linear(self.bond_feature_dim, self.bond_feature_dim)
            self.mol_bond_neighbor_fc = nn.Linear(self.bond_feature_dim, self.bond_feature_dim)
        if self.attn_type == 'GAT':
            self.mol_GAT_align = nn.Linear(2 * self.bond_feature_dim, 1)
        elif self.attn_type == 'GATV2':
            self.mol_GATV2_align = nn.Linear(self.bond_feature_dim, 1)
        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]

        self.layer_norm = nn.LayerNorm(self.bond_feature_dim)
        self.need_layer_norm = cfg.layer_norm

    def forward(self, mol_bond_feature, bond_feature, mol_bond_attend_mask, mol_bond_softmax_mask):
        if self.need_Linear_trans == True:
            mol_bond_feature = self.dropout(mol_bond_feature)
            bond_feature = self.dropout(bond_feature)
            mol_bond_feature = self.mol_bond_fc(mol_bond_feature)
            bond_feature = self.mol_bond_neighbor_fc(bond_feature)

        batch_size, mol_bond_length, bond_feature_dim = bond_feature.shape
        mol_bond_expand = mol_bond_feature.unsqueeze(-2).expand(batch_size, mol_bond_length, bond_feature_dim)

        if self.attn_type == 'GAT':
            mol_feature_align = torch.cat([mol_bond_expand, bond_feature], dim=-1)
            mol_align_score = self.GNN_attn_act(self.mol_GAT_align(mol_feature_align))
            mol_align_score = mol_align_score + mol_bond_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * mol_bond_attend_mask
            mol_bond_context = torch.sum(torch.mul(mol_attention_weight, bond_feature), -2)
        elif self.attn_type == 'GATV2':
            mol_feature_align = mol_bond_expand + bond_feature
            mol_feature_align = self.GNN_attn_act(mol_feature_align)
            mol_align_score = self.mol_GATV2_align(mol_feature_align)
            mol_align_score = mol_align_score + mol_bond_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * mol_bond_attend_mask
            mol_bond_context = torch.sum(torch.mul(mol_attention_weight, bond_feature), -2)
        if self.need_layer_norm == True:
            mol_bond_context = self.layer_norm(mol_bond_context)

        return mol_bond_context

class GNN_mol_atom_update(nn.Module):
    def  __init__(self, atom_feature_dim):
        super().__init__()
        self.update_type = cfg.update_type
        self.atom_feature_dim = atom_feature_dim
        self.GNN_update_act = act_dict[cfg.GNN_update_act]
        if self.update_type == 'skipconcat':
            self.skipconcat_trans = nn.Linear(2 * self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'GRU':
            self.atom_GRUCell = nn.GRUCell(self.atom_feature_dim, self.atom_feature_dim)

    def forward(self, mol_atom_feature, mol_atom_context):
        # mol_atom_state = None
        if self.update_type == 'skipsum':
            mol_atom_state = mol_atom_feature + mol_atom_context
            mol_activated_atom_feature = self.GNN_update_act(mol_atom_state)
        elif self.update_type == 'skipconcat':
            mol_atom_state = torch.cat([mol_atom_feature, mol_atom_context], dim=-1)
            mol_activated_atom_feature = self.GNN_update_act(self.skipconcat_trans(mol_atom_state))
        elif self.update_type == 'GRU':
            mol_atom_state = self.atom_GRUCell(mol_atom_context, mol_atom_feature)
            mol_activated_atom_feature = self.GNN_update_act(mol_atom_state)

        return  mol_activated_atom_feature

class GNN_mol_bond_update(nn.Module):
    def __init__(self, bond_feature_dim):
        super().__init__()
        self.update_type = cfg.update_type
        self.bond_feature_dim = bond_feature_dim
        self.GNN_update_act = act_dict[cfg.GNN_update_act]
        if self.update_type == 'skipconcat':
            self.skipconcat_trans = nn.Linear(2 * self.bond_feature_dim, self.bond_feature_dim)
        elif self.update_type == 'GRU':
            self.bond_GRUCell = nn.GRUCell(self.bond_feature_dim, self.bond_feature_dim)

    def forward(self, mol_bond_feature, mol_bond_context):
        if self.update_type == 'skipsum':
            mol_bond_state = mol_bond_feature + mol_bond_context
            mol_activated_bond_feature = self.GNN_update_act(mol_bond_state)
        elif self.update_type == 'skipconcat':
            mol_bond_state = torch.cat([mol_bond_feature, mol_bond_context], dim=-1)
            mol_activated_bond_feature = self.GNN_update_act(self.skipconcat_trans(mol_bond_state))
        elif self.update_type == 'GRU':
            mol_bond_state = self.bond_GRUCell(mol_bond_context, mol_bond_feature)
            mol_activated_bond_feature = self.GNN_update_act(mol_bond_state)

        return mol_activated_bond_feature


class Frag_atom_cross_attention(nn.Module):
    def __init__(self, atom_feature_dim, need_Linear_trans=True):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.need_Linear_trans = need_Linear_trans
        self.attn_type = cfg.attn_type
        self.dropout = nn.Dropout(cfg.dropout)

        if need_Linear_trans:
            self.mol_atom_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
            self.mol_atom_neighbor_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)

        if self.attn_type == 'GAT':
            self.mol_GAT_align = nn.Linear(2 * self.atom_feature_dim, 1)
        elif self.attn_type == 'GATV2':
            self.mol_GATV2_align = nn.Linear(self.atom_feature_dim, 1)

        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]
        self.need_layer_norm = cfg.layer_norm
        self.layer_norm = nn.LayerNorm(self.atom_feature_dim)

    def forward(self, mol_atom_feature, atom_feature, mol_atom_attend_mask, mol_atom_softmax_mask):
        if mol_atom_attend_mask.ndim == 3 and mol_atom_attend_mask.shape[-1] == 1:
            mol_atom_attend_mask = mol_atom_attend_mask.squeeze(-1)
        if mol_atom_softmax_mask.ndim == 3 and mol_atom_softmax_mask.shape[-1] == 1:
            mol_atom_softmax_mask = mol_atom_softmax_mask.squeeze(-1)
        batch_size, num_agg, atom_feature_dim = mol_atom_feature.shape
        _, mol_atom_length, _ = atom_feature.shape # N_len

        if self.need_Linear_trans:
            mol_atom_feature = self.dropout(mol_atom_feature)
            atom_feature = self.dropout(atom_feature)
            mol_atom_feature = self.mol_atom_fc(mol_atom_feature) # (B, N_agg, D) -> (B, N_agg, D)
            atom_feature = self.mol_atom_neighbor_fc(atom_feature)
        mol_atom_feature_prep = mol_atom_feature.unsqueeze(2) 
        atom_feature_prep = atom_feature.unsqueeze(1) 
        if self.attn_type == 'GAT':
            mol_atom_feature_expanded = mol_atom_feature_prep.expand(-1, -1, mol_atom_length, -1)
            atom_feature_expanded = atom_feature_prep.expand(-1, num_agg, -1, -1)
            mol_feature_align = torch.cat([mol_atom_feature_expanded, atom_feature_expanded], dim=-1)
            mol_align_score = self.mol_GAT_align(mol_feature_align) # (B, N_agg, N_len, 1)
            mol_align_score = self.GNN_attn_act(mol_align_score)

        elif self.attn_type == 'GATV2':
            mol_feature_align = mol_atom_feature_prep + atom_feature_prep # (B, N_agg, N_len, D)
            mol_feature_align = self.GNN_attn_act(mol_feature_align)
            mol_align_score = self.mol_GATV2_align(mol_feature_align)
        mol_atom_softmax_mask_prep = mol_atom_softmax_mask.unsqueeze(1).unsqueeze(-1)
        mol_atom_attend_mask_prep = mol_atom_attend_mask.unsqueeze(1).unsqueeze(-1)
        mol_align_score = mol_align_score + mol_atom_softmax_mask_prep
        mol_attention_weight = F.softmax(mol_align_score, dim=2) # (B, N_agg, N_len, 1)
        mol_attention_weight = mol_attention_weight * mol_atom_attend_mask_prep # (B, N_agg, N_len, 1)
        weighted_values = torch.mul(mol_attention_weight, atom_feature_prep)
        mol_atom_context = torch.sum(weighted_values, dim=2) # (B, N_agg, D)

        if self.need_layer_norm:
            mol_atom_context = self.layer_norm(mol_atom_context) # (B, N_agg, D)

        return mol_atom_context


def info_nce_loss(features, batch_size):
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(cfg.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(cfg.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    logits = logits / cfg.temperature

    loss = criterion(logits, labels)
    return loss

class Layer_attention_by_mol(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_feature_dim = cfg.atom_feature_dim
        self.mol_atom_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        self.mol_atom_neighbor_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        self.mol_layer_GAT_align = nn.Linear(2 * self.atom_feature_dim, 1)

        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, mol_layer_feature_by_atom_sum, layer_neighbor_feature):
        mol_layer_feature_by_atom_sum = self.dropout(mol_layer_feature_by_atom_sum)
        layer_neighbor_feature = self.dropout(layer_neighbor_feature)
        mol_layer_feature_by_atom_sum = self.mol_atom_fc(mol_layer_feature_by_atom_sum)
        layer_neighbor_feature = self.mol_atom_neighbor_fc(layer_neighbor_feature)


        batch_size, layer_nums, mol_atom_fingerprint_dim = layer_neighbor_feature.shape
        mol_layer_feature_by_atom_expand = mol_layer_feature_by_atom_sum.unsqueeze(-2).expand(batch_size, layer_nums,
                                                                                          mol_atom_fingerprint_dim)

        mol_layer_feature_align = torch.cat([mol_layer_feature_by_atom_expand, layer_neighbor_feature], dim=-1)
        mol_layer_align_score = self.GNN_attn_act(self.mol_layer_GAT_align(mol_layer_feature_align))
        mol_layer_attention_weight = F.softmax(mol_layer_align_score, -2)
        mol_layer_atom_context = torch.sum(torch.mul(mol_layer_attention_weight, layer_neighbor_feature), -2)

        return mol_layer_atom_context



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
        self.get_mol_atom_update =GNN_mol_atom_update(cfg.atom_feature_dim)

        self.get_mol_bond_context = GNN_mol_bond_aggregate(self.bond_feature_dim)
        self.get_mol_bond_update = GNN_mol_bond_update(cfg.bond_feature_dim)
        
        self.node_gru = nn.GRU(cfg.atom_feature_dim, cfg.atom_feature_dim, batch_first=True, 
                           bidirectional=True)
        self.frag_input_dim = cfg.atom_dim_in + cfg.bond_dim_in
        self.frag_cross_atten = Frag_atom_cross_attention(self.atom_feature_dim)
        self.frag_input_projection = nn.Linear(self.frag_input_dim, self.atom_feature_dim)

        attn_module = MultiHeadedAttention(h=cfg.frg_nhead, d_model=self.atom_feature_dim, dropout=cfg.dropout)
        ff_module = PositionwiseFeedForward(d_model=self.atom_feature_dim, d_ff=self.atom_feature_dim, dropout=cfg.dropout)
        encoder_layer_no_norm = EncoderLayer(size=self.atom_feature_dim, self_attn=attn_module, feed_forward=ff_module,
                                             dropout=cfg.dropout)
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

        self.descriptor_mlp = nn.ModuleList([GeneralLinearLayer(200,
                                                                cfg.atom_feature_dim,
                                                                has_act=True,
                                                                has_dropout=False,
                                                                need_layernorm=False)]+ [GeneralLinearLayer(cfg.atom_feature_dim,
                                                                                                          cfg.atom_feature_dim,
                                                                                                          has_act=True,
                                                                                                          has_dropout=True,
                                                                                                          need_layernorm=False) for _ in range(cfg.des_num_layers-1)])
        self.postGNN_classifier = PostGNN_classifier(cfg.atom_feature_dim, output_units_num, cfg.classifier_numlayers)
        self.layer_atten = Layer_attention_by_mol()
        self.layer_update_act = act_dict[cfg.GNN_update_act]


    def forward(self, x_atoms, x_bonds, frag_src_features, atom_neighbors_atom_index, atom_neighbors_bond_index,
                    bond_neighbors_bond_index, bond_neighbors_atom_index, atom_mask, bond_mask, frag_src_valid_mask, molecule_descriptor):
        global mol_feature_by_atom, mol_feature_by_bond
        
        
        global op_mol_atom,op_mol_motif,op_mol_descrip
        

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


            atom_neighbor_feature = get_atom_neighbor_feature_atom_bond_new(atom_feature_nb, bond_feature_nb, atom_neighbors_atom_index,
                                                                atom_neighbors_bond_index)
            bond_neighbor_feature = get_bond_neighbor_feature_bond_atom_new(atom_feature_nb, bond_feature_nb, bond_neighbors_bond_index,
                                                                bond_neighbors_atom_index)
            atom_context = torch.sum(torch.mul(atom_attend_mask, atom_neighbor_feature), -2)
            atom_feature_ = self.atom_mlps[i]((1 + self.eps[i]) * atom_feature + atom_context)
            atom_feature = atom_feature_+atom_feature

            bond_context = torch.sum(torch.mul(bond_attend_mask, bond_neighbor_feature), -2)
            bond_feature_ = self.bond_mlps[i]((1 + self.eps[i]) * bond_feature + bond_context)
            bond_feature = bond_feature_+bond_feature
            
            if i==0:
                op_mol_atom = torch.sum(atom_feature * mol_atom_attend_mask, dim=-2)

        frag_features = F.relu(self.frag_input_projection(frag_src_features))
        atom_hidden = atom_feature.max(1)[0].unsqueeze(0).repeat(2,1,1)
        atom_feature, atom_hidden = self.node_gru(atom_feature, atom_hidden)
        
        atom_feature_reshaped = atom_feature.view(batch_size, mol_atom_num, 2, cfg.atom_feature_dim)
        atom_feature = atom_feature_reshaped[:, :, 0, :] + atom_feature_reshaped[:, :, 1, :]
        
        frag_context = self.frag_cross_atten(frag_features, atom_feature,mol_atom_attend_mask, mol_atom_softmax_mask)
        frag_features = frag_features + frag_context

        frag_src_mask = frag_src_valid_mask
        frag_output, last_layer_attn_weights,op_motif = self.encoder_no_norm(frag_features, frag_src_mask)
        frag_cls = frag_output[:, 0, :]
        op_mol_motif = op_motif[:, 0, :]

        mol_feature_by_atom = torch.sum(atom_feature * mol_atom_attend_mask, dim=-2)
        mol_context_by_atom = self.get_mol_atom_context(mol_feature_by_atom, atom_feature, mol_atom_attend_mask,
                                                           mol_atom_softmax_mask)
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
            if i==0:
                op_mol_descrip = molecule_descriptor

        mol_context_by_des = self.get_mol_atom_context_des(molecule_descriptor, atom_feature, mol_atom_attend_mask,
                                                        mol_atom_softmax_mask)
        mol_feature_by_des = self.get_mol_atom_update_des(molecule_descriptor, mol_context_by_des)

        layer_neighbor_feature = torch.stack([frag_cls, mol_feature_by_atom, mol_feature_by_des], dim=1)

        mol_layer_feature_by_atom_sum = torch.sum(layer_neighbor_feature, dim=-2)

        mol_layer_context = self.layer_atten(mol_layer_feature_by_atom_sum, layer_neighbor_feature)
        mol_final_fea = self.layer_update_act(mol_layer_context + mol_layer_feature_by_atom_sum)


        output = self.postGNN_classifier(mol_final_fea)
        return atom_feature, output, ct_loss, op_mol_atom,op_mol_motif,op_mol_descrip,mol_final_fea




