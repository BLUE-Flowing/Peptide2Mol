import pandas as pd
import torch
import os, sys
import pickle
import traceback
from copy import deepcopy
import lmdb
import numpy as np
import pandas as pd
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
    atomic_number = atom.GetAtomicNum()
    periods = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2,
               11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3}
    groups = {1: 1, 2: 8, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
              11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8}
    
    period = periods.get(atomic_number, -1) 
    group = groups.get(atomic_number, -1) 
    return period, group


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def calc_atom_features(atom):
    atom_symbol = [ 'C',  'N',  'O',  'F',  'P',  'S', 'Cl', 'Br']
    atom_degree = [ 0, 1, 2, 3, 4, 5, 6 ]
    hybrid_type = [ Chem.rdchem.HybridizationType.SP,    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,   Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2, 'other'] 

    period, group = get_period_group(atom)

    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbol)        \
            + one_of_k_encoding(atom.GetDegree(), atom_degree)            \
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]     \
            + one_of_k_encoding_unk(atom.GetHybridization(), hybrid_type) \
            + [atom.GetIsAromatic()] \
            + [period, group] 

    
                                        
    return np.array(results)

def print_atom_features(ligand_dict):
    for idx, fea in enumerate(ligand_dict['feature_atoms']):
        atom_symbol = fea[0:8]
        atom_degree = fea[8:15]
        formal_charge = fea[15]
        num_radical_electrons = fea[16]
        hybrid_type = fea[17:23]
        is_aromatic = fea[23]
        peroid = fea[24]
        group = fea[25]

        atom_symbol_str = "Is C: " + ("Yes" if atom_symbol[0] else "Not") + "\n"
        atom_symbol_str += "Is N: " + ("Yes" if atom_symbol[1] else "Not") + "\n"
        atom_symbol_str += "Is O: " + ("Yes" if atom_symbol[2] else "Not") + "\n"
        atom_symbol_str += "Is F: " + ("Yes" if atom_symbol[3] else "Not") + "\n"
        atom_symbol_str += "Is P: " + ("Yes" if atom_symbol[4] else "Not") + "\n"
        atom_symbol_str += "Is S: " + ("Yes" if atom_symbol[5] else "Not") + "\n"
        atom_symbol_str += "Is Cl: " + ("Yes" if atom_symbol[6] else "Not") + "\n"
        atom_symbol_str += "Is Br: " + ("Yes" if atom_symbol[7] else "Not") + "\n"

        degree_str = "Degree: " + str(np.argmax(atom_degree)) + "\n" if 1 in atom_degree else "Degree: Unknown\n"
        formal_charge_str = "FormalCharge: " + str(formal_charge) + "\n"
        num_radical_electrons_str = "NumRadicalElectrons: " + str(num_radical_electrons) + "\n"

        hybrid_type_str = "Is SP: " + ("Yes" if hybrid_type[0] else "Not") + "\n"
        hybrid_type_str += "Is SP2: " + ("Yes" if hybrid_type[1] else "Not") + "\n"
        hybrid_type_str += "Is SP3: " + ("Yes" if hybrid_type[2] else "Not") + "\n"
        hybrid_type_str += "Is SP3D: " + ("Yes" if hybrid_type[3] else "Not") + "\n"
        hybrid_type_str += "Is SP3D2: " + ("Yes" if hybrid_type[4] else "Not") + "\n"
        hybrid_type_str += "Is Other: " + ("Yes" if hybrid_type[5] else "Not") + "\n"

        aromatic_str = "Is Aromatic: " + ("Yes" if is_aromatic else "Not") + "\n"
        peroid_str = "Period: " + str(peroid) + "\n"
        group_str = "Group: " + str(group) + "\n"

        print(f"Atom id {idx}:\n" + atom_symbol_str + degree_str + formal_charge_str + num_radical_electrons_str +
              hybrid_type_str + aromatic_str + peroid_str + group_str)

def print_bond_features(ligand_dict):
    bond_indexs = ligand_dict['bond_index']
    for idx, fea in enumerate(ligand_dict['feature_bonds']):

        atom_bonded_str = "Atom Bonded: " + str(bond_indexs[0][idx].item()) + " and " + str(bond_indexs[1][idx].item()) + "\n"

        single_bond = fea[0]
        double_bond = fea[1]
        triple_bond = fea[2]
        aromatic_bond = fea[3]
        conjugated_bond = fea[4]
        in_ring_bond = fea[5]
        non_cov_in_4_A = fea[6]

        single_bond_str = "Is Single: " + ("Yes" if single_bond else "Not") + "\n"
        double_bond_str = "Is Double: " + ("Yes" if double_bond else "Not") + "\n"
        triple_bond_str = "Is Triple: " + ("Yes" if triple_bond else "Not") + "\n"
        aromatic_bond_str = "Is Aromatic: " + ("Yes" if aromatic_bond else "Not") + "\n"
        conjugated_bond_str = "Is Conjugated: " + ("Yes" if conjugated_bond else "Not") + "\n"
        in_ring_bond_str = "Is InRing: " + ("Yes" if in_ring_bond else "Not") + "\n"
        special_bond_str = "Is non_cov_in_4_A: " + ("Yes" if non_cov_in_4_A else "Not") + "\n"

        print(f"Bond id {idx}:\n" + atom_bonded_str + single_bond_str + double_bond_str + triple_bond_str + aromatic_bond_str +
              conjugated_bond_str + in_ring_bond_str + special_bond_str)

def calc_bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
           bt == Chem.rdchem.BondType.SINGLE, 
           bt == Chem.rdchem.BondType.DOUBLE,
           bt == Chem.rdchem.BondType.TRIPLE, 
           bt == Chem.rdchem.BondType.AROMATIC,
           bond.GetIsConjugated(),
           bond.IsInRing(), 0]

    return np.array(bond_feats).astype(int)

def safe_remove(inp_index, remove_indices_list):
    """
    Renumber the inp_index after removing indices in remove_indices_list from all_indices.

    Parameters:
    inp_index (int): The original index to be renumbered.
    remove_indices_list (list of int): The list of indices to remove.

    Returns:
    int or None: The new index if inp_index is not removed, otherwise None.
    """

    # Check if inp_index is to be removed
    if inp_index in remove_indices_list:
        return None  # Indicate that inp_index has been removed

    # Count how many removed indices are less than inp_index
    removed_before = sum(1 for x in remove_indices_list if x < inp_index)

    # Calculate the new index
    new_index = inp_index - removed_before

    return new_index


def parse_drug3d_mol(mol, diffu_idx, name, remove_idx):
    remove_indices_list = list(map(int, remove_idx.split(';')))
    num_bonds = mol.GetNumBonds()
    num_atoms = mol.GetNumAtoms()
    feature_atoms = np.zeros((num_atoms-len(remove_indices_list), 26))
    bond_feats_all = []
    conf = mol.GetConformer()
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms) if i not in remove_indices_list])
    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        if i not in remove_indices_list:
            pos = conf.GetAtomPosition(i)
            ele = atom.GetAtomicNum()
            pos_list.append(list(pos))
            ele_list.append(ele)

    try:
        Chem.SanitizeMol(mol)
    except:
            try: 	
                Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                pass
    for i in range(num_atoms):
        # print(i, remove_indices_list)
        if i not in remove_indices_list:
            # print()
            atom = mol.GetAtomWithIdx(i)
            feature_atoms[safe_remove(i, remove_indices_list)] = calc_atom_features(atom)
    chiral_arr    = np.zeros([num_atoms-len(remove_indices_list), 3]) 
    chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation=False)
    for (i, rs) in chiralcenters:
        if i not in remove_indices_list:
            if rs == 'R':
                chiral_arr[safe_remove(i, remove_indices_list), 0] =1 
            elif rs == 'S':
                chiral_arr[safe_remove(i, remove_indices_list), 1] =1 
            else:
                chiral_arr[safe_remove(i, remove_indices_list), 2] =1 
    feature_atoms = np.concatenate([feature_atoms, chiral_arr], axis=1)


    dist_matrix = squareform(pdist(positions))

    # Define the cutoff for interaction (4 Å)
    cutoff = 5.0

    # Get existing bonds and initialize graph edges and types
    existing_bonds = set()
    row, col, bond_type = [], [], []
    diff_element, diff_pos, diff_bond_type, diff_bond_type_idx, diff_atom_features, diff_bond_features = [], [], [], [], [], []
    diff_row, diff_col = [], []
    num_diff_bonds = 0
    indices_list = list(map(int, diffu_idx.split(';')))
    # print(indices_list)
    safe_indices_list = []
    for ix in indices_list:
        # print(ix, safe_remove(ix, remove_indices_list))
        if safe_remove(ix, remove_indices_list) != None:
            safe_indices_list.append(safe_remove(ix, remove_indices_list))
            

    max_pocket_atom_id = max(safe_indices_list) # ligand must be in front of receptor, the atoms are removed from ligand

    for idx, bond in enumerate(mol.GetBonds()):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start not in remove_indices_list and end not in remove_indices_list:
            # of note, the atom id has been added (which in remove_indices_list has been removed) so here we can directly add it
            existing_bonds.add((safe_remove(start, remove_indices_list), safe_remove(end, remove_indices_list)))
            existing_bonds.add((safe_remove(end, remove_indices_list), safe_remove(start, remove_indices_list)))
            # Convert RDKit bond type to simple integer if necessary
            b_type = int(bond.GetBondTypeAsDouble())
            bond_feats = calc_bond_features(bond)
            bond_feats_all.append(bond_feats)
            bond_feats_all.append(bond_feats)
            if b_type == 12:
                b_type = 4  # Assuming this maps dative bonds or any special cases
            row.extend([safe_remove(start, remove_indices_list), safe_remove(end, remove_indices_list)])
            col.extend([safe_remove(end, remove_indices_list), safe_remove(start, remove_indices_list)])
            bond_type.extend([b_type, b_type])
            if start in indices_list or end in indices_list:
                num_diff_bonds += 1
                diff_bond_type.extend([b_type, b_type])
                diff_bond_features.append(bond_feats)
                diff_bond_features.append(bond_feats)
                diff_row.extend([safe_remove(start, remove_indices_list), safe_remove(end, remove_indices_list)])
                diff_col.extend([safe_remove(end, remove_indices_list), safe_remove(start, remove_indices_list)])

    pocket_or_not_list = np.zeros([num_atoms - len(remove_indices_list)]) # this list has 

    # Check for non-bonded interactions within cutoff and add them
    for i in range(num_atoms):
        if i not in remove_indices_list:
            ix = safe_remove(i, remove_indices_list)
            if ix > max_pocket_atom_id:
                pocket_or_not_list[ix] = 1

        for j in range(i + 1, num_atoms):
            if j not in remove_indices_list:
                jx = safe_remove(j, remove_indices_list)
                if (ix, jx) not in existing_bonds: 
                    if dist_matrix[ix, jx] < cutoff :
                        row.extend([ix, jx])
                        col.extend([jx, ix])
                        bond_type.extend([5, 5])
                        bond_feats_all.append([0, 0, 0, 0, 0, 0, 1])
                        bond_feats_all.append([0, 0, 0, 0, 0, 0, 1])
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
    

    for i, atom in enumerate(mol.GetAtoms()):
        if i in indices_list and i not in remove_indices_list:
            diff_element.append(atom.GetAtomicNum())
            diff_pos.append(list(conf.GetAtomPosition(i)))
            diff_atom_features.append(feature_atoms[safe_remove(i, remove_indices_list)])
    
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
        'num_atoms': num_atoms - len(remove_indices_list),
        'num_bonds': num_bonds,
        'num_diff_bonds': num_diff_bonds,
        'diffu_idx': safe_indices_list,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_drug3d_dicts(ligand_dict=None, **kwargs):
        instance = Drug3DData(**kwargs)

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance[key] = item
            instance['orig_keys'] = list(ligand_dict.keys())

        # instance['nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
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
    for _, line in tqdm(df_use.iterrows(), total=len(df_use), desc='Preprocessing data'):
        mol_id = line['filename']
        db = lmdb.open(
            os.path.join(processed_path, f'{mol_id[:-4]}.pt'),
            map_size=500*(1024*1024*1024),   # 500GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
        with db.begin(write=True, buffers=True) as txn:
            
            # mol info
            
            diffu_idx = line['diffu_idx']
            remove_idx = line['remove_idx']
            num_skipped = 0
            
            try:
                # load all confs of the mol
                suppl = Chem.SDMolSupplier(os.path.join(sdf_path, mol_id))
                mol = suppl[0]
                # santinize the mol
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveAllHs(mol)
                ligand_dict = parse_drug3d_mol(mol, diffu_idx, mol_id, remove_idx)
                ligand_dict = torchify_dict(ligand_dict)
                data = Drug3DData.from_drug3d_dicts(ligand_dict)
                remove_indices_list = list(map(int, remove_idx.split(';')))
                rw_mol = Chem.RWMol(mol)
                rw_mol = deepcopy(rw_mol)
                remove_indices_sorted = sorted(remove_indices_list, reverse=True)
                for idx in remove_indices_sorted:
                    if idx < 0 or idx >= rw_mol.GetNumAtoms():
                        raise IndexError(f"Atom index {idx} is out of range for molecule {mol_id}")
                    rw_mol.RemoveAtom(idx)
                
                # Get the modified molecule from the writable molecule
                modified_mol = rw_mol.GetMol()
                
                pdb_filename = os.path.join(os.path.join(processed_path, f'{mol_id[:-4]}_remaining.pdb'))
                Chem.MolToPDBFile(modified_mol, pdb_filename)
                print(pdb_filename)

                remove_indices_list.extend(list(map(int, diffu_idx.split(';'))))
                print(remove_indices_list, diffu_idx)
                rw_mol = Chem.RWMol(mol)
                remove_indices_sorted = sorted(remove_indices_list, reverse=True)
                for idx in remove_indices_sorted:
                    if idx < 0 or idx >= rw_mol.GetNumAtoms():
                        raise IndexError(f"Atom index {idx} is out of range for molecule {mol_id}")
                    rw_mol.RemoveAtom(idx)
                
                # Get the modified molecule from the writable molecule
                modified_mol = rw_mol.GetMol()
                pdb_filename = os.path.join(os.path.join(processed_path, f'{mol_id[:-4]}.pdb'))
                Chem.MolToPDBFile(modified_mol, pdb_filename)
                print(pdb_filename)
    
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