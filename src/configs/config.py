import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class Config():

    ##############输入输出文件路径#####################

    hyper_log_file = "./hyper_log_file/BACE_scaffold_split"
    task_name = 'bace'
    tasks = ['Class']
    data_raw_filename = "./data/bace.csv"

    #####################训练超参#####################
    batch_size = 64
    epochs = 100

    learning_rate = 5e-4

    lr_FACTOR = 0.8####0.8
    lr_PATIENCE = 10###15
    lr_MIN_LR = 1e-5

    final_lr =1e-4
    max_epochs = 100
    warmup_epochs = 2
    max_lr = 1e-3

    early_roc_epochs = 50 #15
    early_loss_epochs = 20 #20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr_scheduler_type = 'noam'##'noam'reduce

    #####################模型结构超参#####################
    atom_feature_dim = 256  
    bond_feature_dim = 256  

    hop_coff = 2 

    radius = 2  

    frag_num_encoder_layers = 2

    preGNN_num_layers = 1

    channel_reduction = 2

    #####################模型模块选择超参#####################
    GNN_update_act = 'lrelu'
    GNN_attn_act = 'lrelu'

    atom_attn_type = 'GAT'
    atom_update_type = 'actminGRU'#####################actminGRU
    atom_head_nums = 1

    global_attn_type = 'GAT'
    global_update_type = 'skipsum'
    gloabal_head_nums = 8

    layer_atten_query = 'final_layer

    preGNN_act = 'lrelu'

    frg_nhead = 8

    #####################模型正则化超参#####################
    dropout = 0.3
    layer_norm = False
    weight_decay = 1e-5

    #####################固定变量#####################
    atom_dim_in = 39  
    bond_dim_in = 76  

    per_task_output_units_num = 2

    ctloss_name = "infonce" 


    mol_query_preGNN_num_layers = 1
    mol_layer_query_preGNN_num_layers = 1


    classifier_numlayers = 1
    classifier_reduction = False
    attn_type = 'GAT'
    update_type = 'skipsum'

    ediff_center = np.arange(0, 1.7, 0.2)
    ediff_gamma = 5.0

    bond_len_centers = np.arange(0, 2.3, 0.1)
    bond_len_gamma = 10.0



    # ########################fusion type##########################
    fusion_type = 'sum'

    temperature = 0.15
    ct_loss_coff = 0.15
    
    des_num_layers = 2



cfg = Config()



