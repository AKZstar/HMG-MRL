from hyperopt import Trials, hp
from hyperopt import fmin, tpe, STATUS_OK
import numpy as np
import hyperopt

import torch
import os
import argparse
import numpy as np
import pickle
import warnings
from sklearn.model_selection import KFold
from config import cfg
from other_utils import *
from torch.utils.data import DataLoader

from itertools import product


from preprocess.get_atom_bond_frag_info import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import importlib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
import sys

hyperparameter_setting = {
    'atom_feature_dim': [256],
    'bond_feature_dim': [256],
    'radius': [2],
    'frag_num_encoder_layers':[2],
    'des_num_layers':[2],
    'temperature':[0.15],
    'ct_loss_coff':[0.15],
    'classifier_numlayers':[1],
    'dropout': [0.3],
    'learning_rate': [1e-3],
    'lr_FACTOR':[0.8],
    'lr_PATIENCE':[8],
    'weight_decay': [1e-5],  
    'gloabal_head_nums':[8],
    'seed_number':[2020,2021,2022,2023,2024],

}

def start_main(hyperparameter_setting,model_name):
    current_iteration = 1
    for param_values in product(*hyperparameter_setting.values()):
        param_dict = dict(zip(hyperparameter_setting.keys(), param_values))
        logger = create_logger(name=f"{model_name}_Iteration_{current_iteration}", save_dir=os.path.join(cfg.hyper_log_file, model_name),
log_filename=f"{model_name}_hyperparams_iteration_{current_iteration}.log")
        logger.info(f"{model_name} for iteration {current_iteration} params:\n {param_dict}")
        cfg.seed_number = param_dict['seed_number']
        cfg.atom_feature_dim = param_dict['atom_feature_dim']
        cfg.bond_feature_dim = param_dict['bond_feature_dim']
        cfg.radius = param_dict['radius']
        cfg.frag_num_encoder_layers = param_dict['frag_num_encoder_layers']
        cfg.des_num_layers = param_dict['des_num_layers']
        cfg.temperature = param_dict['temperature']
        cfg.ct_loss_coff = param_dict['ct_loss_coff']
        cfg.classifier_numlayers = param_dict['classifier_numlayers']
        cfg.dropout = param_dict['dropout']
        cfg.learning_rate = param_dict['learning_rate']
        cfg.lr_FACTOR = param_dict['lr_FACTOR']
        cfg.lr_PATIENCE = param_dict['lr_PATIENCE']
        cfg.weight_decay = param_dict['weight_decay']
        cfg.gloabal_head_nums = param_dict['gloabal_head_nums']
        start_pogram(logger)
        current_iteration += 1


def start_pogram(logger):
    seed_everything(cfg.seed_number)
    remained_df_filename = "./data/" + cfg.task_name + "_remained_df.pickle"
    remained_df = pickle.load(open(remained_df_filename, "rb"))

    train_df, valid_df, test_df = scaffold_split_valid_test(remained_df, balanced=True,random_state=cfg.seed_number)
    label_weights = calculate_label_balanced_weight(cfg.tasks, remained_df)
    loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight), reduction='mean') for weight in label_weights]
    output_units_num = cfg.per_task_output_units_num * len(cfg.tasks)
    model = GNN_atom_bond(cfg.atom_dim_in, cfg.bond_dim_in, output_units_num)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate, weight_decay= cfg.weight_decay)

    if cfg.lr_scheduler_type == 'noam':
        scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[cfg.warmup_epochs], total_epochs=[cfg.max_epochs],
                       steps_per_epoch= (len(train_df) + cfg.batch_size - 1) // cfg.batch_size,
                       init_lr=[1e-4], max_lr=[cfg.max_lr], final_lr=[cfg.final_lr])
    elif cfg.lr_scheduler_type == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.lr_FACTOR,
            patience=cfg.lr_PATIENCE,
            min_lr=cfg.lr_MIN_LR
        )
    initialize_weights(model)
    print_model_pm(model)
    feature_filename = "./data/" + cfg.task_name + ".pickle"
    feature_dicts = pickle.load(open(feature_filename, "rb"))
    final_result = train_and_val(loss_function, model, optimizer, remained_df, feature_dicts, logger, scheduler)
    return final_result

def train_and_val(loss_function, model, optimizer, remained_df, feature_dicts, logger, scheduler):
    train_df, valid_df, test_df = scaffold_split_valid_test(remained_df, balanced=True,random_state=cfg.seed_number)
    
    train_df.to_csv(os.path.join("./data/")+f'fold_{cfg.seed_number-2020}_train.csv',index=False)
    valid_df.to_csv(os.path.join("./data/")+f'fold_{cfg.seed_number-2020}_valid.csv',index=False)
    test_df.to_csv(os.path.join("./data/")+f'fold_{cfg.seed_number-2020}_test.csv',index=False)

    best_param = {}
    best_param["roc_epoch"] = 0
    best_param["loss_epoch"] = 0
    best_param["valid_roc"] = 0
    best_param["valid_loss"] = 9e8

    best_param["best_epoch_train_metric"] = 0
    best_param["best_epoch_test_metric"] = 0

    best_epoch_valid = 0
    best_epoch_train = 0

    for epoch in range(cfg.epochs):

        logger.info(('#' * 10 + "The Training epoch[{:0>4}/{:0>4}] has started!" + '#' * 10).format(
            epoch, cfg.epochs))

        train_roc_, train_loss_ = train(loss_function, model, optimizer, train_df, feature_dicts, logger, epoch, scheduler)

        train_roc, train_loss = eval(loss_function, model, optimizer, train_df, feature_dicts, logger)
        valid_roc, valid_loss = eval(loss_function, model, optimizer, valid_df, feature_dicts, logger)
        test_roc, test_loss = eval(loss_function, model, optimizer, test_df, feature_dicts, logger)

        train_roc_mean = np.array(train_roc).mean()
        valid_roc_mean = np.array(valid_roc).mean()
        test_roc_mean = np.array(test_roc).mean()

        if valid_roc_mean >= best_param["valid_roc"]:
            best_param["roc_epoch"] = epoch
            best_param["valid_roc"] = valid_roc_mean
            best_param["best_epoch_train_metric"] = train_roc_mean
            best_param["best_epoch_test_metric"] = test_roc_mean
            
            torch.save(model, 'saved_models/model_' + '_' + str(epoch) + '.pt')
        if valid_loss < best_param["valid_loss"]:
            best_param["loss_epoch"] = epoch
            best_param["valid_loss"] = valid_loss
         
        if epoch==80 or epoch==90:
            torch.save(model, 'saved_models/model_' + '_' + str(epoch) + '.pt')
        logger.info(('#' * 10 + "The Training epoch[{:0>4}/{:0>4}] has been finished!" + '#' * 10).format(
            epoch, cfg.epochs))

        logger.info("learning rate {:.8f},"
                    .format(optimizer.param_groups[0]['lr']))

        logger.info("epoch training loss : {:.4f}, epoch training mereic : {:.4f}, "
                    .format(train_loss, train_roc_mean))

        logger.info("epoch valid loss : {:.4f}, epoch valid mereic : {:.4f}, "
                    .format(valid_loss, valid_roc_mean))

        logger.info("epoch test loss : {:.4f}, epoch test mereic : {:.4f}, "
                    .format(test_loss, test_roc_mean))

        logger.info("best roc epoch： {}, best train roc: {:.4f}, best valid roc: {:.4f}, best test roc: {:.4f}, best loss epoch： {}"
                    .format(best_param["roc_epoch"], best_param["best_epoch_train_metric"], best_param["valid_roc"], best_param["best_epoch_test_metric"], best_param["loss_epoch"]))


        if (epoch - best_param["roc_epoch"] > cfg.early_roc_epochs) and (epoch - best_param["loss_epoch"] > cfg.early_loss_epochs):
            break

        if cfg.lr_scheduler_type == 'reduce':
            scheduler.step(valid_loss)





def train(loss_function, model, optimizer, dataset, feature_dicts, logger, epoch, scheduler):
    model.train()

    y_val_list = {}
    y_pred_list = {}
    losses_list = []

    np.random.seed(cfg.seed_number+epoch)
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


        x_atoms, x_bonds, frag_features, atom_neighbors_atom_index, atom_neighbors_bond_index, bond_neighbors_bond, \
            bond_neighbors_atom, atom_mask, bond_mask, frag_mask, smiles_to_frag_atom_bond_incices,molecule_descriptor,_, _ = get_smiles_array(smiles_list, feature_dicts)



        atoms_prediction, mol_prediction,ct_loss, op_mol_atom,op_mol_motif,op_mol_descrip,mol_final_fea = model(torch.Tensor(x_atoms), torch.Tensor(x_bonds),
                                                          torch.Tensor(frag_features),
                                                          torch.cuda.LongTensor(atom_neighbors_atom_index),
                                                          torch.cuda.LongTensor(atom_neighbors_bond_index),
                                                          torch.cuda.LongTensor(bond_neighbors_bond),
                                                          torch.cuda.LongTensor(bond_neighbors_atom),
                                                          torch.Tensor(atom_mask),
                                                          torch.Tensor(bond_mask),
                                                          torch.Tensor(frag_mask),
                                                 torch.Tensor(molecule_descriptor))

        optimizer.zero_grad()
        loss = 0.0
        for i, task in enumerate(cfg.tasks):
            y_pred = mol_prediction[:, i * cfg.per_task_output_units_num:(i + 1) *
                                                                     cfg.per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))

            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]

            y_val_list[i].extend(y_val_adjust)
            y_pred_list[i].extend(y_pred_adjust)
        loss =  loss + cfg.ct_loss_coff * ct_loss

        loss.backward()
        optimizer.step()

        batch_eval_loss = (loss / len(cfg.tasks)).cpu().detach().numpy()
        losses_list.append(batch_eval_loss)

        logger.info(
            "### Training process: Iteration[{:0>3}/{:0>3}], Loss: {:.4f}, op learing_rate: {:.8f} ###".format(
                counter + 1, len(batch_list),
                batch_eval_loss,optimizer.param_groups[0]['lr']#, scheduler.get_last_lr()[0]
            ))
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
        batch_df = dataset.loc[eval_batch, :]
        smiles_list = batch_df.cano_smiles.values
        x_atoms, x_bonds, frag_features, atom_neighbors_atom_index, atom_neighbors_bond_index, bond_neighbors_bond, \
            bond_neighbors_atom, atom_mask, bond_mask, frag_mask, smiles_to_frag_atom_bond_incices, molecule_descriptor,_, _ = get_smiles_array(
            smiles_list, feature_dicts)
        # atoms_prediction, mol_prediction = model(torch.Tensor(x_atoms), torch.Tensor(x_bonds),torch.cuda.LongTensor(atom_neighbors_atom_index), torch.cuda.LongTensor(atom_neighbors_bond_index),torch.Tensor(atom_mask))
        atoms_prediction, mol_prediction, ct_loss, op_mol_atom,op_mol_motif,op_mol_descrip,mol_final_fea = model(torch.Tensor(x_atoms), torch.Tensor(x_bonds), torch.Tensor(frag_features),
                                                 torch.cuda.LongTensor(atom_neighbors_atom_index),
                                                 torch.cuda.LongTensor(atom_neighbors_bond_index),
                                                 torch.cuda.LongTensor(bond_neighbors_bond),
                                                 torch.cuda.LongTensor(bond_neighbors_atom),
                                                 torch.Tensor(atom_mask),
                                                 torch.Tensor(bond_mask),
                                                 torch.Tensor(frag_mask),
                                                 torch.Tensor(molecule_descriptor))

        for i, task in enumerate(cfg.tasks):
            y_pred = mol_prediction[:, i * cfg.per_task_output_units_num:(i + 1) *
                                                                     cfg.per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
            #             print(y_pred_adjust)
            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            losses_list.append(loss.cpu().detach().numpy())
            batch_eval_loss.append(loss.cpu().detach().numpy())
            y_val_list[i].extend(y_val_adjust)
            y_pred_list[i].extend(y_pred_adjust)
        loss_batch.append((sum(batch_eval_loss)+cfg.ct_loss_coff*(ct_loss.cpu().detach().numpy())))
        
    eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(cfg.tasks))]
    eval_loss = np.array(loss_batch).mean()

    return eval_roc, eval_loss  

def import_model(model_name):
    module_name = f'model.{model_name}'
    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"模型模块 {module_name} 未找到！")
        return None
    globals().update(model_module.__dict__)
    return model_module

if __name__ == '__main__':
    models = [
        "bipartite_transformer_knowledge_head_0",
        
    ] 

    for model_name in models:
        print(f"开始运行模型 {model_name} 的训练、验证和测试...")
        import_model(model_name)
        start_main(hyperparameter_setting, model_name)
        print(f"模型 {model_name} 的训练、验证和测试完成。\n")
