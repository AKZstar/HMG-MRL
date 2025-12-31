# HMG-MRL
Source code for HMG-MRL: Hierarchical Multi-Granularity Molecular Representation Learning.
# Requirements
The environment configuration required to run this project can be found in the environment.yml file.
# Files
## data
We use nine benchmark datasets for molecular property prediction, including six classification datasets—BACE, BBBP, SIDER, ClinTox, HIV, and Tox21—and three regression datasets—ESOL, FreeSolv, and Lipophilicity.
All datasets are obtained from MoleculeNet (https://moleculenet.org/datasets-1).
## preprocess
Electronegativity.pkl: This file stores the Pauling electronegativity values of all atoms, which are used for extracting the bond feature based on electronegativity differences.
fg_dicts.txt: This file stores various functional groups and their corresponding SMARTS patterns, which are used to identify functional groups in a given molecule.
Featurizer_atom_bond.py: This code file is used to extract initial atom and bond features based on RDKit.
motif_utils.py: This code file extracts diverse molecular motifs based on three substructure partition strategies, including BRICS fragmentation, scaffold decomposition, and functional group matching.
get_atom_bond_frag_info.py: This code file models molecules as graph-structured data and extracts various molecular representations, including the initial atom feature matrix, bond feature matrix, and motif feature matrix, as well as molecular descriptors. It also generates the neighbor index matrices for atoms and bonds, along with the RDKit index lists for atoms and bonds.
preprocess_prog.ipynb: A data preprocessing notebook that executes the complete data cleaning pipeline, including SMILES canonicalization, deduplication, and outlier filtering. It calls the feature extraction modules to generate molecular graph data and outputs two pickle files: a feature dictionary (bace.pickle) and the cleaned dataset (bace_remained_df.pickle).
## model
act_func.py: An activation function registry that defines multiple activation functions for flexible use across different model layers.
bipartite_transformer.py: This file provides a complete implementation of the proposed HMG-MRL model.
layer_utils.py: A utility library for neural network layers, providing reusable low-level model components.

#code
other_utils.py: This file contains helper functions used during training, such as scaffold splitting and random seed management.
config.py: This file implements a Config class for centralized management of all project parameters, such as model hyperparameters and training configurations.
run_main.py: The main entry point of the project, responsible for the complete workflow of model training, validation, and testing.

# Quick Start

### 1. Data Preprocessing
jupyter notebook preprocess/preprocess_prog.ipynb
### Execute all cells to generate:
- data/bace.pickle (feature dictionary)
- data/bace_remained_df.pickle (cleaned dataset)
### 2. Model Training
python run_main.py
**Note:** To switch datasets, modify the following lines in `config.py`:
task_name = 'bace'                    # Dataset name
tasks = ['Class']                     # Task label column(s)
data_raw_filename = "./data/bace.csv" # Data file path### 2. Model Training
