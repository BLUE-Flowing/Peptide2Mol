"""
====================================================================================
Script: Main Data Preprocess for peptide2mol dataset
Project: Peptide2Mol
Author: Xinheng He, Yijia Zhang, et al.
Description:
    This script converts molecular structures from SDF files into a PyTorch
    Geometric-friendly dataset (LMDB format) for downstream use in diffusion-based
    molecule generation or peptide-to-small-molecule modeling.

    Each molecule is parsed into atomic and bond features, geometric coordinates,
    and optionally diffusion indices (representing the peptide-derived atoms).

    The final output is a serialized dataset stored in LMDB format for efficient
    random access during training or evaluation.
====================================================================================
"""

import os
import sys
import pickle
import traceback
import lmdb
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform

def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

def get_period_group(atom):
    """
    Get the periodic table period and group for a given atom.
    (Supports elements up to Argon; others return -1.)
    """
    atomic_number = atom.GetAtomicNum()
    periods = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2,
               11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3}
    groups = {1: 1, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
              11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8}
    
    period = periods.get(atomic_number, -1) 
    group = groups.get(atomic_number, -1) 
    return period, group


def one_of_k_encoding(x, allowable_set):
    """
    Return a one-hot encoding of an input value given an allowable set.
    Raises an error if the input is not in the set.
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def calc_atom_features(atom):
    """ Currently encodes atom type among ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'] """
    atom_symbol = [ 'C',  'N',  'O',  'F',  'P',  'S', 'Cl', 'Br']
    period, group = get_period_group(atom)
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbol)                              
    return np.array(results)

def calc_bond_features(bond):
    """
    Compute bond type features as a binary vector:
    [is_single, is_double, is_triple, is_aromatic, is_special]
    """
    bt = bond.GetBondType()
    bond_feats = [
           bt == Chem.rdchem.BondType.SINGLE, 
           bt == Chem.rdchem.BondType.DOUBLE,
           bt == Chem.rdchem.BondType.TRIPLE, 
           bt == Chem.rdchem.BondType.AROMATIC,
             0]
    return np.array(bond_feats).astype(int)

def parse_drug3d_mol(mol, diffu_idx, name, cutoff=5.):
    """
    Parse an RDKit Mol object into atom, bond, and geometric features.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        diffu_idx (str): Semicolon-separated indices of atoms involved in diffusion.
        name (str): Molecule identifier.
        cutoff (float): Pocket atom distance cutoff threshold

    Returns:
        dict: Contains all features required for PyTorch Geometric graph construction.
    """
    num_bonds = mol.GetNumBonds()
    num_atoms = mol.GetNumAtoms()
    feature_atoms = np.zeros((num_atoms, 8))
    bond_feats_all = []
    conf = mol.GetConformer()
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])

    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        ele = atom.GetAtomicNum()
        pos_list.append(list(pos))
        ele_list.append(ele)
        
    # Sanitize molecule to ensure chemical validity
    try:
        Chem.SanitizeMol(mol)
    except:
            try: 	
                Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                pass
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        feature_atoms[i] = calc_atom_features(atom)
    dist_matrix = squareform(pdist(positions))
    
    # Get existing bonds and initialize graph edges and types
    existing_bonds = set()
    row, col, bond_type = [], [], []
    diff_element, diff_pos, diff_bond_type, diff_bond_type_idx, diff_atom_features, diff_bond_features = [], [], [], [], [], []
    diff_row, diff_col = [], []
    num_diff_bonds = 0
    indices_list = list(map(int, diffu_idx.split(';')))
    max_pocket_atom_id = max(indices_list)
    
    # Build edge list for existing bonds
    for idx, bond in enumerate(mol.GetBonds()):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        existing_bonds.add((start, end))
        existing_bonds.add((end, start))
        # Convert RDKit bond type to simple integer if necessary
        b_type = int(bond.GetBondTypeAsDouble())
        bond_feats = calc_bond_features(bond)
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)
        if b_type == 12:
            b_type = 4  # Assuming this maps dative bonds or any special cases
        row.extend([start, end])
        col.extend([end, start])
        bond_type.extend([b_type, b_type])
        if start in indices_list or end in indices_list:
            num_diff_bonds += 1
            diff_bond_type.extend([b_type, b_type])
            diff_bond_features.append(bond_feats)
            diff_bond_features.append(bond_feats)
            diff_row.extend([start, end])
            diff_col.extend([end, start])
            
    # Assign pocket vs. non-pocket atoms
    pocket_or_not_list = np.zeros([num_atoms]) 

    # Check for non-bonded interactions within cutoff and add them
    for i in range(num_atoms):
        if i > max_pocket_atom_id:
            pocket_or_not_list[i] = 1

        for j in range(i + 1, num_atoms):
            if (i, j) not in existing_bonds: 
                if dist_matrix[i, j] < cutoff :
                    row.extend([i, j])
                    col.extend([j, i])
                    bond_type.extend([5, 5])
                    bond_feats_all.append([0, 0, 0, 0, 1])
                    bond_feats_all.append([0, 0, 0, 0, 1])
                    num_bonds += 1  
    # Prepare final arrays
    bond_index = np.array([row, col], dtype=np.int64)
    bond_type = np.array(bond_type, dtype=np.int64)
    bond_feats_all = np.array(bond_feats_all, dtype=np.int64)

    # Sort by bond index for consistency
    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    bond_index = bond_index[:, perm]
    bond_type = bond_type[perm]
    bond_feats_all = bond_feats_all[perm]
    
    # Prepare diffusion-specific features
    for i, atom in enumerate(mol.GetAtoms()):
        if i in indices_list:
            diff_element.append(atom.GetAtomicNum())
            diff_pos.append(list(conf.GetAtomPosition(i)))
            diff_atom_features.append(feature_atoms[i])
    
    diff_bond_type = np.array(diff_bond_type, dtype=np.int64)
    diff_bond_index = np.array([diff_row, diff_col],dtype=np.int64)
    diff_bond_features = np.array(diff_bond_features, dtype=np.int64)

    perm = (diff_bond_index[0] * num_atoms + diff_bond_index[1]).argsort() 
    diff_bond_index = diff_bond_index[:, perm]
    diff_bond_type = diff_bond_type[perm]
    diff_bond_features = diff_bond_features[perm]

    for index, pairs in enumerate(bond_index.T):
        if pairs[0] in indices_list or pairs[1] in indices_list:
            # print(index, pairs, 'index, pairs')
            diff_bond_type_idx.append(index)

    data = {
        'name': name,
        'element': np.array(ele_list, dtype=np.int64),
        'pocket_or_not': np.array(pocket_or_not_list, dtype=np.int64), 
        'feature_atoms': feature_atoms,
        'feature_bonds': bond_feats_all,
        'pos': np.array(pos_list, dtype=np.float32),
        'bond_index': np.array(bond_index, dtype=np.int64),
        'bond_type': np.array(bond_type, dtype=np.int64),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'num_diff_bonds': num_diff_bonds,
        'diffu_idx': indices_list,
        'diff_element': np.array(diff_element, dtype=np.int64),
        'diff_pos': np.array(diff_pos, dtype=np.float32),
        'diff_bond_index': np.array(diff_bond_index, dtype=np.int64),
        'diff_bond_type': np.array(diff_bond_type, dtype=np.int64),
        'diff_bond_type_idx': np.array(diff_bond_type_idx, dtype=np.int64),
        'diff_bond_features': np.array(diff_bond_features, dtype=np.int64),
        'diff_atom_features': np.array(diff_atom_features, dtype=np.int64),
    }
    return data

class Drug3DData(Data):
    """
    PyTorch Geometric Data subclass for 3D molecule graphs.
    Supports LMDB serialization and custom indexing logic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_drug3d_dicts(ligand_dict=None, **kwargs):
        instance = Drug3DData(**kwargs)

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance[key] = item
            instance['orig_keys'] = list(ligand_dict.keys())
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bond_index':
            return len(self['node_type'])
        elif key == 'edge_index':
            return len(self['node_type'])
        elif key == 'halfedge_index':
            return len(self['node_type'])
        else:
            return super().__inc__(key, value, *args, **kwargs)

def main():
    df_use = pd.read_csv(sys.argv[1])
    sdf_path = sys.argv[2]
    processed_path = sys.argv[3]

    db = lmdb.open(
                processed_path,
                map_size=500*(1024*1024*1024),   # 500GB
                create=True,
                subdir=False,
                readonly=False, # Writable
            )
    
    num_skipped = 0
    with db.begin(write=True, buffers=True) as txn:
        for _, line in tqdm(df_use.iterrows(), total=len(df_use), desc='Preprocessing data'):
            # mol info
            mol_id = line['filename']
            diffu_idx = line['diffu_idx']
            try:
                # load all confs of the mol
                suppl = Chem.SDMolSupplier(os.path.join(sdf_path, mol_id))
                mol = suppl[0]
                # santinize the mol
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveAllHs(mol)
                ligand_dict = parse_drug3d_mol(mol, diffu_idx, mol_id)
                ligand_dict = torchify_dict(ligand_dict)
                data = Drug3DData.from_drug3d_dicts(ligand_dict)

                data.mol_id = mol_id+str(_)
                
                txn.put(
                    key = str(mol_id+str(_)).encode(),
                    value = pickle.dumps(data)
                )
            except:
                traceback.print_exc()
                num_skipped += 1
                print('Skipping (%d) Num: %s' % (num_skipped, mol_id))
                # assert False
                continue
    db.close()
    print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)

if __name__ == '__main__':
    main()
