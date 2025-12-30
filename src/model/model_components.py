import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from config import cfg
from act_func import act_dict
from layer_utils import *


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


def attention(query, key, value, mask=None, dropout=None):
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
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_1(x).relu())


class SublayerConnection(nn.Module):
   
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(x))


class EncoderLayer(nn.Module):
    
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


class Encoder(nn.Module):
    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        op_mol_motif = None
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, mask)
            
            if i == 0:
                op_mol_motif = x
            
            if i == len(self.layers) - 1:
                last_layer_attn_weights = attn_weights
        return x, last_layer_attn_weights, op_mol_motif



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
    
    def __init__(self, atom_feature_dim):
        super().__init__()
        self.update_type = cfg.update_type
        self.atom_feature_dim = atom_feature_dim
        self.GNN_update_act = act_dict[cfg.GNN_update_act]
        
        if self.update_type == 'skipconcat':
            self.skipconcat_trans = nn.Linear(2 * self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'GRU':
            self.atom_GRUCell = nn.GRUCell(self.atom_feature_dim, self.atom_feature_dim)

    def forward(self, mol_atom_feature, mol_atom_context):
        if self.update_type == 'skipsum':
            mol_atom_state = mol_atom_feature + mol_atom_context
            mol_activated_atom_feature = self.GNN_update_act(mol_atom_state)
        elif self.update_type == 'skipconcat':
            mol_atom_state = torch.cat([mol_atom_feature, mol_atom_context], dim=-1)
            mol_activated_atom_feature = self.GNN_update_act(self.skipconcat_trans(mol_atom_state))
        elif self.update_type == 'GRU':
            mol_atom_state = self.atom_GRUCell(mol_atom_context, mol_atom_feature)
            mol_activated_atom_feature = self.GNN_update_act(mol_atom_state)

        return mol_activated_atom_feature


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
        _, mol_atom_length, _ = atom_feature.shape
        if self.need_Linear_trans:
            mol_atom_feature = self.dropout(mol_atom_feature)
            atom_feature = self.dropout(atom_feature)
            mol_atom_feature = self.mol_atom_fc(mol_atom_feature)
            atom_feature = self.mol_atom_neighbor_fc(atom_feature)
        mol_atom_feature_prep = mol_atom_feature.unsqueeze(2) 
        atom_feature_prep = atom_feature.unsqueeze(1) 

        if self.attn_type == 'GAT':
            mol_atom_feature_expanded = mol_atom_feature_prep.expand(-1, -1, mol_atom_length, -1)
            atom_feature_expanded = atom_feature_prep.expand(-1, num_agg, 1, -1)
            mol_feature_align = torch.cat([mol_atom_feature_expanded, atom_feature_expanded], dim=-1)
            mol_align_score = self.mol_GAT_align(mol_feature_align)
            mol_align_score = self.GNN_attn_act(mol_align_score)

        elif self.attn_type == 'GATV2':
            mol_feature_align = mol_atom_feature_prep + atom_feature_prep
            mol_feature_align = self.GNN_attn_act(mol_feature_align)
            mol_align_score = self.mol_GATV2_align(mol_feature_align)

        mol_atom_softmax_mask_prep = mol_atom_softmax_mask.unsqueeze(1).unsqueeze(-1)
        mol_atom_attend_mask_prep = mol_atom_attend_mask.unsqueeze(-1).unsqueeze(-1)
        
        mol_align_score = mol_align_score + mol_atom_softmax_mask_prep
        mol_attention_weight = F.softmax(mol_align_score, dim=2)
        mol_attention_weight = mol_attention_weight * mol_atom_attend_mask_prep

        weighted_values = torch.mul(mol_attention_weight, atom_feature_prep)
        mol_atom_context = torch.sum(weighted_values, dim=2)

        if self.need_layer_norm:
            mol_atom_context = self.layer_norm(mol_atom_context)

        return mol_atom_context



def info_nce_loss(features, batch_size):
    """InfoNCE对比学习损失函数"""
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
        mol_layer_feature_by_atom_expand = mol_layer_feature_by_atom_sum.unsqueeze(-2).expand(
            batch_size, layer_nums, mol_atom_fingerprint_dim)

        mol_layer_feature_align = torch.cat([mol_layer_feature_by_atom_expand, layer_neighbor_feature], dim=-1)
        mol_layer_align_score = self.GNN_attn_act(self.mol_layer_GAT_align(mol_layer_feature_align))
        mol_layer_attention_weight = F.softmax(mol_layer_align_score, -2)
        mol_layer_atom_context = torch.sum(torch.mul(mol_layer_attention_weight, layer_neighbor_feature), -2)

        return mol_layer_atom_context