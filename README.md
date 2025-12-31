# HMG-MRL

Source code for **HMG-MRL: Hierarchical Multi-Granularity Molecular Representation Learning**.

---

## Requirements

The environment configuration required to run this project can be found in the `environment.yml` file.

---

## Files

### data

We use nine benchmark datasets for molecular property prediction, including six classification datasets (BACE, BBBP, SIDER, ClinTox, HIV, and Tox21) and three regression datasets (ESOL, FreeSolv, and Lipophilicity).

All datasets are obtained from [MoleculeNet](https://moleculenet.org/datasets-1).

---

### preprocess

- **Electronegativity.pkl**  
  Stores the Pauling electronegativity values of all atoms, used for extracting bond features based on electronegativity differences.

- **fg_dicts.txt**  
  Contains various functional groups and their corresponding SMARTS patterns for functional group identification.

- **Featurizer_atom_bond.py**  
  Extracts initial atom and bond features based on RDKit.

- **motif_utils.py**  
  Extracts diverse molecular motifs using three substructure partition strategies: BRICS fragmentation, scaffold decomposition, and functional group matching.

- **get_atom_bond_frag_info.py**  
  Models molecules as graph-structured data and extracts atom-, bond-, and motif-level feature matrices, molecular descriptors, neighbor index matrices, and RDKit index lists.

- **preprocess_prog.ipynb**  
  A data preprocessing notebook that performs SMILES canonicalization, deduplication, and outlier filtering.  
  It generates two pickle files:
  - `bace.pickle` (feature dictionary)
  - `bace_remained_df.pickle` (cleaned dataset)

---

### model

- **act_func.py**  
  An activation function registry for flexible selection across different model layers.

- **bipartite_transformer.py**  
  Provides a complete implementation of the proposed HMG-MRL model.

- **layer_utils.py**  
  A utility library for reusable low-level neural network components.

---

### code

- **other_utils.py**  
  Provides auxiliary utilities for training, including scaffold-based data splitting and random seed control.

- **config.py**  
  Implements a `Config` class for centralized management of all project parameters, including model hyperparameters and training settings.

- **run_main.py**  
  The main entry point of the project, responsible for the complete workflow of model training, validation, and testing.

---

## Quick Start

### 1. Data Preprocessing

Run the following notebook:

```bash
jupyter notebook preprocess/preprocess_prog.ipynb
