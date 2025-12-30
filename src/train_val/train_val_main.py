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

from get_atom_bond_features_with_bond_len_ediff_with_frag_0515 import *

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

# 导入拆分出的模块
from train_config import SEARCH_SPACE, grid_search_hyperparam_main
from train_eval_utils import train, eval


def train_and_val(loss_function, model, optimizer, remained_df, feature_dicts, logger, scheduler):
    train_df, valid_df, test_df = scaffold_split_valid_test(
        remained_df, balanced=True, 
        random_state=2020, 
    )
    
    train_df.to_csv(os.path.join("./data/") + f'128_fold_train.csv', index=False)
    valid_df.to_csv(os.path.join("./data/") + f'128_fold_valid.csv', index=False)
    test_df.to_csv(os.path.join("./data/") + f'128_fold_test.csv', index=False)

    best_param = {}
    best_param["roc_epoch"] = 0
    best_param["loss_epoch"] = 0
    best_param["valid_roc"] = 0
    best_param["valid_loss"] = 9e8
    best_param["best_epoch_train_metric"] = 0
    best_param["best_epoch_test_metric"] = 0

    for epoch in range(cfg.epochs):
        logger.info(('#' * 10 + "The Training epoch[{:0>4}/{:0>4}] has started!" + '#' * 10).format(
            epoch, cfg.epochs))

        train_roc_, train_loss_ = train(loss_function, model, optimizer, train_df, 
                                        feature_dicts, logger, epoch, scheduler)

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
        
        if epoch == 80 or epoch == 90:
            torch.save(model, 'saved_models/model_' + '_' + str(epoch) + '.pt')

        logger.info(('#' * 10 + "The Training epoch[{:0>4}/{:0>4}] has been finished!" + '#' * 10).format(
            epoch, cfg.epochs))

        logger.info("learning rate {:.8f},".format(optimizer.param_groups[0]['lr']))

        logger.info("epoch training loss : {:.4f}, epoch training metric : {:.4f}, ".format(
            train_loss, train_roc_mean))

        logger.info("epoch valid loss : {:.4f}, epoch valid metric : {:.4f}, ".format(
            valid_loss, valid_roc_mean))

        logger.info("epoch test loss : {:.4f}, epoch test metric : {:.4f}, ".format(
            test_loss, test_roc_mean))

        logger.info("best roc epoch: {}, best train roc: {:.4f}, best valid roc: {:.4f}, best test roc: {:.4f}, best loss epoch: {}".format(
            best_param["roc_epoch"], best_param["best_epoch_train_metric"], 
            best_param["valid_roc"], best_param["best_epoch_test_metric"], 
            best_param["loss_epoch"]))

        if (epoch - best_param["roc_epoch"] > cfg.early_roc_epochs) and \
           (epoch - best_param["loss_epoch"] > cfg.early_loss_epochs):
            logger.info("Early stopping triggered!")
            break

        if cfg.lr_scheduler_type == 'reduce':
            scheduler.step(valid_loss)

    return best_param


def import_model(model_name):
    module_name = f'backbond_modified_model.{model_name}'
    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"模型模块 {module_name} 未找到！")
        return None

    globals().update(model_module.__dict__)
    return model_module


if __name__ == '__main__':
    models = [
        "bipartite_transformer_knowledge",
    ] 

    for model_name in models:
        print(f"开始运行模型 {model_name} 的训练、验证和测试...")
        import_model(model_name)

        grid_search_hyperparam_main(SEARCH_SPACE, model_name)
        print(f"模型 {model_name} 的训练、验证和测试完成。\n")