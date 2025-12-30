"""
训练和评估工具模块
包含训练和评估的核心逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from config import cfg
from get_atom_bond_features_with_bond_len_ediff_with_frag_0515 import get_smiles_array


def train(loss_function, model, optimizer, dataset, feature_dicts, logger, epoch, scheduler):
    model.train()

    y_val_list = {}
    y_pred_list = {}
    losses_list = []

    np.random.seed(cfg.seed_number + epoch)
    valList = np.arange(0, dataset.shape[0])
    np.random.shuffle(valList)
    
    batch_list = []
    for i in range(len(cfg.tasks)):
        y_val_list[i] = []
        y_pred_list[i] = []
    for i in range(0, dataset.shape[0], cfg.batch_size):
        batch = valList[i:i + cfg.batch_size]
        batch_list.append(batch)

    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atoms, x_bonds, frag_features, atom_neighbors_atom_index, atom_neighbors_bond_index, \
        bond_neighbors_bond, bond_neighbors_atom, atom_mask, bond_mask, frag_mask, \
        smiles_to_frag_atom_bond_incices, molecule_descriptor, _, _ = get_smiles_array(
            smiles_list, feature_dicts
        )

        atoms_prediction, mol_prediction, ct_loss, op_mol_atom, op_mol_motif, op_mol_descrip, mol_final_fea = model(
            torch.Tensor(x_atoms), 
            torch.Tensor(x_bonds),
            torch.Tensor(frag_features),
            torch.cuda.LongTensor(atom_neighbors_atom_index),
            torch.cuda.LongTensor(atom_neighbors_bond_index),
            torch.cuda.LongTensor(bond_neighbors_bond),
            torch.cuda.LongTensor(bond_neighbors_atom),
            torch.Tensor(atom_mask),
            torch.Tensor(bond_mask),
            torch.Tensor(frag_mask),
            torch.Tensor(molecule_descriptor)
        )

        optimizer.zero_grad()
        loss = 0.0
        for i, task in enumerate(cfg.tasks):
            y_pred = mol_prediction[:, i * cfg.per_task_output_units_num:(i + 1) * cfg.per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](y_pred_adjust, torch.cuda.LongTensor(y_val_adjust))

            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            y_val_list[i].extend(y_val_adjust)
            y_pred_list[i].extend(y_pred_adjust)
        
        loss = loss + cfg.ct_loss_coff * ct_loss
        
        loss.backward()
        optimizer.step()

        batch_eval_loss = (loss / len(cfg.tasks)).cpu().detach().numpy()
        losses_list.append(batch_eval_loss)

        logger.info(
            "### Training process: Iteration[{:0>3}/{:0>3}], Loss: {:.4f}, op learing_rate: {:.8f} ###".format(
                counter + 1, len(batch_list), batch_eval_loss, optimizer.param_groups[0]['lr']
            )
        )
        
        if cfg.lr_scheduler_type == 'noam':
            scheduler.step()

    eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(cfg.tasks))]
    eval_loss = np.array(losses_list).mean()

    return eval_roc, eval_loss


def eval(loss_function, model, optimizer, dataset, feature_dicts, logger):
    model.eval()
    
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    loss_batch = []
    
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    
    for i in range(len(cfg.tasks)):
        y_val_list[i] = []
        y_pred_list[i] = []
    
    for i in range(0, dataset.shape[0], cfg.batch_size):
        batch = valList[i:i + cfg.batch_size]
        batch_list.append(batch)
    
    for counter, eval_batch in enumerate(batch_list):
        batch_eval_roc = []
        batch_eval_loss = []

        batch_df = dataset.loc[eval_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atoms, x_bonds, frag_features, atom_neighbors_atom_index, atom_neighbors_bond_index, \
        bond_neighbors_bond, bond_neighbors_atom, atom_mask, bond_mask, frag_mask, \
        smiles_to_frag_atom_bond_incices, molecule_descriptor, _, _ = get_smiles_array(
            smiles_list, feature_dicts
        )

        atoms_prediction, mol_prediction, ct_loss, op_mol_atom, op_mol_motif, op_mol_descrip, mol_final_fea = model(
            torch.Tensor(x_atoms), 
            torch.Tensor(x_bonds), 
            torch.Tensor(frag_features),
            torch.cuda.LongTensor(atom_neighbors_atom_index),
            torch.cuda.LongTensor(atom_neighbors_bond_index),
            torch.cuda.LongTensor(bond_neighbors_bond),
            torch.cuda.LongTensor(bond_neighbors_atom),
            torch.Tensor(atom_mask),
            torch.Tensor(bond_mask),
            torch.Tensor(frag_mask),
            torch.Tensor(molecule_descriptor)
        )

        for i, task in enumerate(cfg.tasks):
            y_pred = mol_prediction[:, i * cfg.per_task_output_units_num:(i + 1) * cfg.per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            
            loss = loss_function[i](y_pred_adjust, torch.cuda.LongTensor(y_val_adjust))
            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            losses_list.append(loss.cpu().detach().numpy())

            batch_eval_loss.append(loss.cpu().detach().numpy())
            y_val_list[i].extend(y_val_adjust)
            y_pred_list[i].extend(y_pred_adjust)
        
        loss_batch.append((sum(batch_eval_loss) + cfg.ct_loss_coff * (ct_loss.cpu().detach().numpy())))

    eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(cfg.tasks))]
    eval_loss = np.array(loss_batch).mean()

    return eval_roc, eval_loss