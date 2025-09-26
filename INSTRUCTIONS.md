# FragNet instructions for EGFR

This document provides instructions for using FragNet to analyze the Epidermal Growth Factor Receptor (EGFR) protein.

## Basic environment setup

Notices: Only execute if you have not set up the environment before.

```bash 
git clone https://github.com/pnnl/FragNet.git
cd FragNet && pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
cd FragNet && pip install .
```

## Usage on EGFR data

### Notices before running

#### `exp`

- Likely means experimental conformations.
- Uses real 3D coordinates from crystallographic datasets (e.g., PDBBind, QM9 with DFT geometries).
- Best if you have actual 3D structures available.

#### `exp1`

- Variant of exp, usually means:
    - Generate 1 conformation per molecule using a toolkit (e.g., RDKit ETKDG).
    - Useful when experimental 3D structures are not available.
- This gives each molecule one canonical 3D geometry.

#### `exp1s`

- The “s” usually stands for symmetric / multiple seeds / stochastic.
- Often means:
    - Generate multiple conformations per molecule (e.g., 5–10 random conformers).
    - Then either pick one at random or use all for augmentation.
- Ensures more robust training when real 3D data is missing.

### Pretrain EGFR

```bash
python data_create/create_pretrain_datasets.py \
--save_path pretrain_data/egfr \
--data_type exp1s \
--maxiters 500 \
--raw_data_path finetune_data/mylab/egfr/raw/20250905_egfr_train_test_data.csv

# python data_create/create_pretrain_datasets.py \
# --save_path pretrain_data/egfr \
# --data_type exp1s \
# --maxiters 500 \
# --raw_data_path finetune_data/moleculenet/egfr/raw/20250905_egfr_train_test_data.csv
```

### Finetune EGFR

```bash
python data_create/create_finetune_datasets.py \
--dataset_name mylab \
--dataset_subset egfr \
--use_molebert True \
--output_dir finetune_data/mylab_egfr \
--data_dir finetune_data/mylab \
--data_type exp1s
```

## Model usage with EGFR data

### Pretrain model

```bash
python train/pretrain/pretrain_gat2.py \
--config exps/pt/unimol_exp1s4/config.yaml
```

### Finetune model

```bash
python train/finetune/finetune_gat2.py \
--config exps/ft/esol/e1pt4.yaml
```