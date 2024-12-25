# Peptide2Mol: A Diffusion Model for Transforming Peptide Binders into Small Molecules

Peptide2Mol is a diffusion-based method for generating small-molecule candidates from peptide binders in drug design. This repository provides the necessary code, instructions, and model weights for inference or retraining. If you have any questions, feel free to [open an issue](https://github.com/Xinheng-He/peptide2mol/issues) or reach out via email at [he-xinheng@foxmail.com](mailto:he-xinheng@foxmail.com).

---

## Table of Contents
1. [Dataset](#dataset)
2. [Setup Environment](#setup-environment)
3. [Running Peptide2Mol on a Test System](#running-peptide2mol-on-a-test-system)
4. [Retraining Peptide2Mol](#retraining-peptide2mol)
5. [License](#license)

---

## Dataset

To train the main Peptide2Mol model, you can download either the processed or original data from Zenodo.  
- If you have the raw data in the `raw_data` folder, you can convert it into the necessary `.pt` format by running:  
  ```bash
  python ./notebooks/deal_with_mol_5A.py ./raw_data/final_csv_goodH.csv ./raw_data/sdf ./data_all5.pt
  ```

---

## Setup Environment

Below is an example of how to set up an Anaconda environment to run Peptide2Mol. Make sure to install the correct versions of PyTorch, PyTorch-Geometric, CUDA (if you have a GPU), etc. You can also adjust versions as needed if you prefer newer releases:

```bash
git clone https://github.com/Xinheng-He/peptide2mol
cd peptide2mol

conda create --name peptide2mol python=3.9.19
conda activate peptide2mol

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 rootutils==1.0.7 pre-commit==3.7.0 rich==13.7.1 pytest==8.1.1
pip install lightning==2.2.2
pip install numpy==1.22.4
pip install pandas==2.2.2
pip install lmdb==1.4.1
pip install rdkit==2023.9.6
pip install torch-geometric==2.5.2
pip install numba==0.59.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install easydict==1.13
pip install tensorboard==2.16.2
```

> **Note:** Installation can take up to one hour on a typical Linux machine.

---

## Running Peptide2Mol on a Test System

1. **Prepare Protein Inputs**  
   Your protein input should be a `.pdb` file containing residues within 6 Å of the peptide ligand (i.e., if any heavy atom of a residue lies within 6 Å of any heavy atom of the ligand, include the entire residue).  
   - In PyMOL, for example, you can use:
     ```bash
     select sel_poc, br. (sele around 5)
     ```
     and then save `sel_poc` as `xxx_poc.pdb` in a folder like `./poc_test/`.

2. **Generate `.pt` Files**  
   Run the following to generate `.pt` files for further generation:
   ```bash
   python ./notebooks/deal_with_mol_test_from_pdb_pal_5A.py ./poc_test/
   ```

3. **Inference**  
   - **Without guidance:**
     ```bash
     python src/eval.py experiment=mol_test \
        ckpt_path=$PWD/ckpts/PMT_major.ckpt \
        ++paths.data_dir=$PWD/poc_test/ \
        +data.lmdb_fn=5wbj_MTOR_poc.pt \
        data=mol_test_true \
        model=Moldiff_test \
        data.infer_batch_size=2 \
        trainer.devices=1 \
        ++paths.log_dir=./logs_1223_test \
        ++model.net.sample.log_dir=$PWD/sample_test \
        ++model.net.sample.pdb_dir=$PWD/poc_test \
        ++model.net.sample.batch_size=10
     ```
   - **With guidance:**
     ```bash
     python src/eval.py experiment=mol_test_gui \
        ckpt_path=$PWD/ckpts/PMT_major.ckpt \
        ++paths.data_dir=$PWD/poc_test/ \
        +data.lmdb_fn=5wbj_MTOR_poc.pt \
        data=mol_test_true \
        model=Moldiff_gui_comp \
        data.infer_batch_size=2 \
        trainer.devices=1 \
        ++paths.log_dir=./logs_1224_test \
        ++model.net.sample.log_dir=$PWD/sample_test \
        ++model.net.sample.pdb_dir=$PWD/poc_test \
        ++model.net.sample.batch_size=1 \
        ++model.net.sample.gui_dir=$PWD/ckpts/PMT_comparison.ckpt
     ```

> **Note:** Generating one molecule typically takes about one minute on an H800 GPU.

---

## Retraining Peptide2Mol

Use `src/train.py` to retrain the model. If your data is located in `../data_all5.pt`, you can run:

```bash
python src/train.py experiment=mol_test \
  ++paths.data_dir=$PWD/../ \
  +data.lmdb_fn=data_all5.pt \
  +data.num_train=370000 \
  ++data.batch_size=8 \
  ++paths.log_dir=./logs_retrain \
  logger=many_loggers
```

---

## License

[MIT](./LICENSE)

---  

*Thank you for using Peptide2Mol! If you have any questions or encounter any issues, please let us know.*
