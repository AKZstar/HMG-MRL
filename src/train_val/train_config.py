"""
训练配置和超参数管理模块
包含超参数搜索空间定义和初始化逻辑
"""

import torch
import os
import numpy as np
import pickle
from itertools import product
from config import cfg
from other_utils import *



SEARCH_SPACE = {
    'atom_feature_dim': [256],
    'bond_feature_dim': [256],
    'radius': [2],
    'frag_num_encoder_layers': [2],
    'des_num_layers': [2],
    'temperature': [0.15],
    'ct_loss_coff': [0.15],
    'classifier_numlayers': [1],
    'dropout': [0.3],
    'learning_rate': [1e-3],
    'lr_FACTOR': [0.8],
    'lr_PATIENCE': [8],
    'weight_decay': [1e-5],
    'gloabal_head_nums': [8],
}

def initialize_experiment(param_dict, logger):
    
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
    
    log_motif_config(logger)


def log_motif_config(logger):
    methods_enabled = []
    if cfg.use_brics_fragments:
        methods_enabled.append("BRICS")
    if cfg.use_murcko_fragments:
        methods_enabled.append("Murcko")
    if cfg.use_smarts_fragments:
        methods_enabled.append("SMARTS")
    
    logger.info("=" * 80)
    logger.info("MOTIF EXTRACTION CONFIGURATION")
    logger.info(f"  use_brics_fragments: {cfg.use_brics_fragments}")
    logger.info(f"  use_murcko_fragments: {cfg.use_murcko_fragments}")
    logger.info(f"  use_smarts_fragments: {cfg.use_smarts_fragments}")
    logger.info(f"  Enabled methods: {', '.join(methods_enabled) if methods_enabled else 'None (only whole_mol)'}")
    logger.info(f"  Config name: {cfg.fragment_methods_config}")
    logger.info("=" * 80)


def prepare_data_and_model(logger):
    from torch import nn
    from model.bipartite_transformer_knowledge import GNN_atom_bond

    remained_df_filename = "./data/" + cfg.task_name + "_remained_df.pickle"
    remained_df = pickle.load(open(remained_df_filename, "rb"))

    label_weights = calculate_label_balanced_weight(cfg.tasks, remained_df)
    loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight), reduction='mean') 
                     for weight in label_weights]
 
    output_units_num = cfg.per_task_output_units_num * len(cfg.tasks)
    model = GNN_atom_bond(cfg.atom_dim_in, cfg.bond_dim_in, output_units_num)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, 
                                 weight_decay=cfg.weight_decay)
    
    from utils.other_utils import scaffold_split_valid_test
    train_df, valid_df, test_df = scaffold_split_valid_test(
        remained_df, balanced=True, 
        random_state_1=2020
    )
    
    if cfg.lr_scheduler_type == 'noam':
        scheduler = NoamLR(
            optimizer=optimizer, 
            warmup_epochs=[cfg.warmup_epochs], 
            total_epochs=[cfg.max_epochs],
            steps_per_epoch=(len(train_df) + cfg.batch_size - 1) // cfg.batch_size,
            init_lr=[1e-4], 
            max_lr=[cfg.max_lr], 
            final_lr=[cfg.final_lr]
        )
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

    feature_filename = "./data/" + cfg.task_name + "_no_smarts" + ".pickle"
    feature_dicts = pickle.load(open(feature_filename, "rb"))
    
    return model, optimizer, scheduler, loss_function, feature_dicts, remained_df


def grid_search_hyperparam_main(search_space, model_name):
    from train_val_main import train_and_val
    
    current_iteration = 1
    for param_values in product(*search_space.values()):
        param_dict = dict(zip(search_space.keys(), param_values))

        logger = create_logger(
            name=f"{model_name}_Iteration_{current_iteration}", 
            save_dir=os.path.join(cfg.hyper_log_file, model_name),
            log_filename=f"{model_name}_hyperparams_iteration_{current_iteration}.log"
        )

        logger.info(f"{model_name} for iteration {current_iteration} params:\n {param_dict}")

        initialize_experiment(param_dict, logger)

        model, optimizer, scheduler, loss_function, feature_dicts, remained_df = prepare_data_and_model(logger)

        final_result = train_and_val(
            loss_function, model, optimizer, remained_df, 
            feature_dicts, logger, scheduler
        )
        
        current_iteration += 1