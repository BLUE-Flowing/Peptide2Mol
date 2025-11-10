# Peptide2Mol: A Diffusion Model for Generating Small Molecules as Peptide Mimics for Targeted Protein Binding

**Peptide2Mol** is a diffusion-based method for generating small-molecule candidates from peptide binders in drug design.  
This repository provides the necessary code, instructions, and model weights for inference or retraining.  
For any questions or issues, feel free to [open an issue](https://github.com/BLUE-Flowing/Peptide2Mol/issues) or reach out via email at [he-xinheng@foxmail.com](mailto:he-xinheng@foxmail.com).

<p align="center">
  <img src="assets/overview.png" alt="Overview of Peptide2Mol" width="600"/>
</p>

<p align="center">
  </b> Overview of the Peptide2Mol diffusion-based framework.
</p>

---
## Quick Links

- [Setup Environment](#setup-environment)
- [Dataset (Optional)](#optional-dataset)
  - [Downloading](#downloading)
  - [Preprocessing](#preprocessing)
- [Training Weights](#training-weights)
- [Data Preparation](#data-preparation)
  - [Step1 Prepare Protein Inputs](#step-1-prepare-protein-inputs)
  - [Step2 Generate .pt Files](#step-2-generate-pt-files)
- [Inference](#inference)
  - [Step3 Demo Generation](#step-3-demo-generation)
- [Step4 Fix Molecules with Pocket2Mol (Optional)](#optional-step-4-fix-molecules-with-pocket2mol)
- [Retraining Peptide2Mol](#retraining-peptide2mol)
- [License](#license)
- [Arxiv Submission](#arxiv-submission)
---

## Setup Environment

Follow these steps to set up an Anaconda environment for running Peptide2Mol. Ensure compatibility by installing the specified versions of **PyTorch, PyTorch-Geometric, CUDA (if applicable)**, and other dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/BLUE-Flowing/Peptide2Mol.git
cd Peptide2Mol

# 2. Create the Conda environment from the provided YAML file
conda env create -f env_cu121.yaml
conda activate peptide2mol

# 3. Install PyTorch-Geometric dependencies for CUDA 12.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

> **Note:** The env_cu121.yaml file is pre-configured for Python 3.9 and CUDA 12.1. If you are using a different GPU or CUDA version, adjust the YAML file or PyTorch-Geometric wheel URL accordingly.

---

## (Optional) Dataset

### Downloading
To train the **Peptide2Mol** model from scratch, you can download the original dataset from **Google Drive**:
[Download the dataset (Drive folder)](https://drive.google.com/drive/folders/1I2uyFPSfeDS1ZXzKxw4ZQ5cu74sQGtSD?hl=zh)

After downloading, you should have the following files:
- **dataset.tar.gz** —— compressed dataset containing structure files
- **final_csv_goodH.csv** —— CSV file containing metadata and diffusion indices

Place both files into the **Peptide2Mol/** directory and then extract the dataset:
```bash
mkdir -p dataset
tar -xzvf dataset.tar.gz -C dataset
mv final_csv_goodH.csv dataset/
```

---

### Preprocessing
After downloading and extracting the dataset into the **`dataset/`** folder, run the preprocessing script to convert the SDF files into PyTorch `.pt` files for model training:
```bash
python ./notebooks/deal_with_mol_5A.py \
       ./dataset/final_csv_goodH.csv \
       ./dataset/sdf_noH \
       ./dataset/data.pt
```
Then, the processed dataset file data.pt will be saved in the **dataset/** directory.
If you plan to retrain the model, make sure the path to data.pt is correct.
> **Note:** Preprocessing may take up to three hours.

---

## Training Weights

**The pretrained model weights** can be downloaded from the [release page](https://github.com/BLUE-Flowing/Peptide2Mol/releases/tag/v1.0).  

These checkpoints were trained on the released dataset described above.

After downloading, place the checkpoint file in the following directory: **'./ckpts/PMT_major.ckpt'**
```bash
mkdir -p ckpts
mv PMT_major.ckpt ckpts/
```

---

## Usage

### Data Preparation

#### Step 1: Prepare Protein Inputs

To construct the receptor pocket model, extract residues located within 6 Å of the peptide ligand from the protein complex structure.  
This can be achieved in **PyMOL** using the following commands:

```bash
select sel_poc, br. (sele around 6)
save xxx_poc.pdb, sel_poc  # xxx can be set as the PDB ID for inference
```

The resulting file xxx_poc.pdb contains all residues within a 6 Å radius of the peptide and can be stored in a designated directory (e.g., **./poc_test/**) for subsequent modeling or analysis.

#### Step 2: Generate `.pt` Files

There are two independent workflows for generating `.pt` files:

##### Workflow A. Generate `.pt` Files for All PDB Files Containing 'poc' in Their Filename

Run the following command to read all PDB files with 'poc' in their filename and generate `.pt` files. Ensure that the corresponding SDF files share the same suffix name:

```bash
python ./notebooks/deal_with_mol_test_from_pdb_pal_5A.py ./demo/example/
```

This will process all PDB files in the `./demo/example/` directory containing 'poc' in their filenames, along with the matching SDF files, and generate the corresponding `.pt` files.

##### Workflow B. Generate `.pt` Files for Partial Ligand Parts

If you wish to generate `.pt` files for only part of the ligand, follow these steps:

1. **Combine the Pocket PDB and the Ligand into One SDF File**  
   First, combine the pocket PDB and ligand into a single SDF file.

2. **Prepare the CSV File**  
   Next, create a CSV file (e.g., `./demo/example/partial_input.csv`) with the following format:

   ```csv
   filename,diffu_idx,remove_idx
   PBmol_2qtg.sdf,0;1;2;3;4;5;6;7;8;9,10;11;12;13;14;15;16;17;18;19
   ```

   - `diffu_idx`: The indices of the ligand atoms that will be kept (all atoms except those to be removed).
   - `remove_idx`: The indices of the atoms that will be removed during diffusion.

3. **Run the Command for `.pt` File Generation**  
   Use the following command to generate the `.pt` files for the partial ligand:

   ```bash
   python ./notebooks/deal_with_mol_remove_5A.py ./demo/example/partial_input.csv ./demo/example ./demo/example
   ```

   - The first `./demo/example` specifies the path to the SDF files.
   - The second `./demo/example` specifies the output path where the `.pt` files will be saved.
   
   Adjust the paths according to your needs.

---

### Inference

#### Step 3: Demo Generation

To facilitate **quick verification** of the software by reviewers, we provide a minimal test dataset. This dataset contains a small set of example protein–peptide complexes that can be used to run the full pipeline end-to-end. Specifically, the dataset includes two protein complex structures:
  - **1bvr** – suitable for de novo small molecule generation
  - **PBMol_2qtg** – suitable for partial generation

These structures are sourced from the LiGAN 10-testcase benchmark [DOI: https://doi.org/10.1039/D1SC05976A] and BioLip2 [DOI: https://doi.org/10.1093/nar/gkad630], respectively. The corresponding .pt files have been pre-generated following the procedure described in Step 2. Users can directly use these preprocessed files to test the pipeline without additional preprocessing.

#### De Novo Generation:

To perform de novo generation, run the following command:
```bash
DATA_DIR=./demo/example MODEL=Moldiff_test LMDB_FILE=1bvr_poc.pt SAMPLE_OUTPUT_DIR=./output/1bvr_poc bash scripts/inference.sh
```

  - **Key Parameters:**
    - DATA_DIR: directory containing the pre-generated .pt files
    - MODEL: specifies the model to use (Moldiff_test)
    - LMDB_FILE: the specific .pt file for the protein complex (1bvr_poc.pt)
    - SAMPLE_OUTPUT_DIR: directory where the generated results will be saved

The generated results will be stored in **`./output/1bvr_poc_SDF`**, including:

 - **100** generated small molecules
 - Original pocket structures corresponding to the protein target

This allows reviewers to quickly verify the **de novo generation workflow**.

#### Partial generation:

To perform partial generation (e.g., ignoring some peptide structure to strengthing ), run the following command:
```bash
DATA_DIR=./demo/example MODEL=Moldiff_test_partial LMDB_FILE=PBmol_2qtg.pt SAMPLE_OUTPUT_DIR=./output/PBmol_2qtg bash scripts/inference.sh
```

  - **Key Parameters:**
    - DATA_DIR: directory containing the pre-generated .pt files
    - MODEL: specifies the model to use (Moldiff_test_partial)
    - LMDB_FILE: the specific .pt file for the protein complex (PBmol_2qtg.pt)
    - SAMPLE_OUTPUT_DIR: directory where the generated results will be saved
   
The generated results will be stored in **`./output/PBmol_2qtg_SDF`**, including:
  - **100** generated small molecules, partially conditioned on the input fragment
  - Original pocket structures corresponding to the protein target


### (Optional) Step 4: Fix Molecules with Pocket2Mol

To refine the molecular structures generated by Peptide2Mol, we use Pocket2Mol [DOI: 
https://doi.org/10.48550/arXiv.2205.07249], a structure-based generative model that reconstructs and optimizes ligand conformations within protein binding pockets.
Pocket2Mol corrects atomic inconsistencies and ensures the generated molecules remain chemically valid and geometrically compatible with the target pocket.

1. Make sure your generated molecules from Peptide2Mol are organized in folders ending with **`_SDF`**, e.g.:
```bash
   ./output/1bvr_poc_SDF/
```

2. **Pocket2Mol pretrained model weights** are available for download from **Google Drive**: [Download Link (Drive folder)](https://drive.google.com/drive/folders/1KfdOczjUPITPhIvCuBmnj4xFTV-iI2xB?usp=sharing), For detailed configuration and usage instructions, please refer to **`./Pocket2Mol/ckpt/README.md`**.

After downloading, place the checkpoint file **Pretrained_Pocket2Mol.pt** into the directory:
```bash
   ./Pocket2Mol/ckpt/
```

3. To run Pocket2Mol and fix incorrect atoms, execute the following command. For instance, to process only the folders containing the keyword "1bvr" in their names (you can change this keyword as needed):

```bash
cd Pocket2Mol
python get_wrong_atom_index.py ../output/ 1bvr
```

Each molecule takes approximately one minute to process on an NVIDIA A100 GPU, and the corrected structures will be saved in  `../output_fixed`.
---

## Retraining Peptide2Mol

To retrain the main Peptide2Mol model, use the following command structure:

```bash
DATA_DIR=./dataset EXPERIMENT=mol_test LMDB_FILE=data.pt NUM_TRAIN=370000 LOG_DIR=./logs_retrain bash scripts/train.sh
```

 - **Key Parameters:**
   - `DATA_DIR`: Path to the folder containing your training `.pt` or LMDB data files.
   - `LMDB_FILE`: The dataset file to use for training (e.g., `data.pt`).
   - `NUM_TRAIN`: Number of training samples (adjust based on your dataset)
   - `LOG_DIR`: Directory to save training logs and checkpoints

 - **General Tips:**
   - Ensure your data files are in the correct directory structure
   - Adjust batch sizes according to your GPU memory capacity
   - Monitor training progress through logs in specified `log_dir`
   - Use absolute paths if running from different directories
   - Consider using nohup or TMUX for long training sessions
---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Arxiv Submission

### Preprint
This work has been made publicly available as a preprint on arXiv:  
[Peptide2Mol: A Diffusion Model for Generating Small Molecules as Peptide Mimics for Targeted Protein Binding](https://arxiv.org/abs/2511.04984)

### Citation

If you find this work useful, please cite:

```bibtex
@article{he2025peptide2mol,
  title     = {Peptide2Mol: A Diffusion Model for Generating Small Molecules as Peptide Mimics for Targeted Protein Binding},
  author    = {He, Xinheng and Zhang, Yijia and Lin, Haowei and Peng, Xingang and Kong, Xiangzhe and Li, Mingyu and Ma, Jianzhu},
  journal   = {arXiv preprint arXiv:2511.04984},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.04984}
}
```

---

Thank you for using **Peptide2Mol**! If you have any questions or encounter any issues, please don't hesitate to reach out.
