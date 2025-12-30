import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
# torch.manual_seed(8) # for reproduce

import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my own modules
# from AttentiveFP.get_atom_bond_features_0_0612 import save_smiles_dicts, get_smiles_dicts, get_smiles_array

from get_atom_bond_features_with_bond_len_ediff_with_frag_0515 import save_smiles_dicts, get_smiles_dicts, get_smiles_array

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

# from rdkit.Chem import rdMolDescriptors, MolSurf
# from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
# from rdkit.Chem import AllChem
from rdkit.Chem import QED
# %matplotlib inline
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import seaborn as sns; sns.set(color_codes=True)

from config import cfg

def preprocessing(listname):
    canonical_smiles_new_list_name = listname + ".pickle"
    canonical_smiles_new_list = pickle.load(open(canonical_smiles_new_list_name, "rb"))
    
    raw_filename = cfg.data_raw_filename###数据存放地址
    filename = raw_filename.replace('.csv','')
    feature_dicts = save_smiles_dicts(canonical_smiles_new_list,filename)

if __name__ == '__main__':
    # seed_everything(cfg.seed_number)
    preprocessing('canonical_smiles_new_list')