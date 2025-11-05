"""Module to check intermolecular distances between ligand and protein."""
# TEST find minimize_distance_to_protein test
from __future__ import annotations

from typing import Any
from typing import Iterable
import numpy as np
import pandas as pd
from rdkit.Chem.rdchem import GetPeriodicTable, Mol, Atom
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def read_sdf_file(sdf_path: str) -> Mol:
    supplier = Chem.SDMolSupplier(sdf_path)
    mols = [mol for mol in supplier if mol is not None]
    if mols:
        return mols[0]  # Assuming you want the first molecule in the SDF file
    else:
        raise ValueError("No valid molecules found in the SDF file.")

def read_pdb_file(pdb_path: str) -> Mol:
    mol = Chem.MolFromPDBFile(pdb_path)
    if mol is None:
        raise ValueError("Failed to read PDB file.")
    return mol



_inorganic_cofactor_elements = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Cl",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Br",
    "Rb",
    "Mo",
    "Cd",
}
_inorganic_cofactor_ccd_codes = {
    "FES",
    "MOS",
    "PO3",
    "PO4",
    "PPK",
    "SO3",
    "SO4",
    "VO4",
}
_water_ccd_codes = {
    "HOH",
}



def _keep_atom(  # noqa: PLR0913, PLR0911
    atom: Atom, ignore_h: bool, ignore_protein: bool, ignore_org_cof: bool, ignore_inorg_cof: bool, ignore_water: bool
) -> bool:
    """Whether to keep atom for given ignore flags."""
    symbol = atom.GetSymbol()
    if ignore_h and symbol == "H":
        return False

    if ignore_inorg_cof and symbol in _inorganic_cofactor_elements:
        return False

    # if loaded from PDB file, we can use the residue names and the hetero flag
    info = atom.GetPDBResidueInfo()
    if info is None:
        if ignore_org_cof:
            return False
        return True

    is_hetero = info.GetIsHeteroAtom()
    if ignore_protein and not is_hetero:
        return False

    residue_name = info.GetResidueName()
    if ignore_water and residue_name in _water_ccd_codes:
        return False

    if ignore_inorg_cof and residue_name in _inorganic_cofactor_ccd_codes:
        return False

    return True

def get_atom_type_mask(mol: Mol, ignore_types: Iterable[str]) -> list[bool]:
    """Get mask for atoms to keep."""
    ignore_types = set(ignore_types)
    unsupported = ignore_types - {"hydrogens", "protein", "organic_cofactors", "inorganic_cofactors", "waters"}
    if unsupported:
        raise ValueError(f"Ignore types {unsupported} not supported.")

    ignore_h = "hydrogens" in ignore_types
    ignore_protein = "protein" in ignore_types
    ignore_org_cof = "organic_cofactors" in ignore_types
    ignore_inorg_cof = "inorganic_cofactors" in ignore_types
    ignore_water = "waters" in ignore_types

    return [
        _keep_atom(a, ignore_h, ignore_protein, ignore_org_cof, ignore_inorg_cof, ignore_water) for a in mol.GetAtoms()
    ]


_periodic_table = GetPeriodicTable()


def check_intermolecular_distance(  # noqa: PLR0913
    mol_pred: Mol,
    mol_cond: Mol,
    radius_type: str = "vdw",
    radius_scale: float = 1.0,
    clash_cutoff: float = 0.75,
    ignore_types: set[str] = {"hydrogens"},
    max_distance: float = 5.0,
    search_distance: float = 6.0,
) -> dict[str, Any]:
    """Check that predicted molecule is not too close and not too far away from conditioning molecule.

    Args:
        mol_pred: Predicted molecule (docked ligand) with one conformer.
        mol_cond: Conditioning molecule (protein) with one conformer.
        radius_type: Type of atomic radius to use. Possible values are "vdw" (van der Waals) and "covalent".
            Defaults to "vdw".
        radius_scale: Scaling factor for the atomic radii. Defaults to 0.8.
        clash_cutoff: Threshold for how much the atoms may overlap before a clash is reported. Defaults
            to 0.05.
        ignore_types: Which types of atoms to ignore in mol_cond. Possible values to include are "hydrogens", "protein",
            "organic_cofactors", "inorganic_cofactors", "waters". Defaults to {"hydrogens"}.
        max_distance: Maximum distance (in Angstrom) predicted and conditioning molecule may be apart to be considered
            as valid. Defaults to 5.0.

    Returns:
        PoseBusters results dictionary.
    """
    coords_ligand = mol_pred.GetConformer().GetPositions()
    coords_protein = mol_cond.GetConformer().GetPositions()

    atoms_ligand = np.array([a.GetSymbol() for a in mol_pred.GetAtoms()])
    atoms_protein_all = np.array([a.GetSymbol() for a in mol_cond.GetAtoms()])

    idxs_ligand = np.array([a.GetIdx() for a in mol_pred.GetAtoms()])
    idxs_protein = np.array([a.GetIdx() for a in mol_cond.GetAtoms()])

    mask = [a.GetSymbol() != "H" for a in mol_pred.GetAtoms()] # mask out not hydrogens atoms
    coords_ligand = coords_ligand[mask, :]
    atoms_ligand = atoms_ligand[mask]
    mask_ligand_idxs = idxs_ligand[mask]
    if ignore_types: # clean out protein hydrogens atoms
        mask = get_atom_type_mask(mol_cond, ignore_types)
        coords_protein = coords_protein[mask, :]
        atoms_protein_all = atoms_protein_all[mask]
        mask_protein_idxs = idxs_protein[mask]

    # get radii
    radius_ligand = _get_radii(atoms_ligand, radius_type) # based on atom type, get each atom radius
    radius_protein_all = _get_radii(atoms_protein_all, radius_type)

    # select atoms that are close to ligand to check for clash
    distances_all = _pairwise_distance(coords_ligand, coords_protein) 
    mask_protein = distances_all.min(axis=0) <= search_distance #* minimize distance for each protein atom with every ligand atom
    distances = distances_all[:, mask_protein] #* distances between ligand and protein atoms that are close to ligand
    radius_protein = radius_protein_all[mask_protein]
    atoms_protein = atoms_protein_all[mask_protein]
    mask_protein_idxs = mask_protein_idxs[mask_protein]

    radius_sum = radius_ligand[:, None] + radius_protein[None, :] #* radius between every selected ligand atom and protein atom
    relative_distance = distances / radius_sum
    violations = relative_distance < 1 / radius_scale #* if relative distance is less than 1/radius_scale, then there might be a clash

    if distances.size > 0:
        distance_to_radius_add_cutoff = distances - radius_sum * clash_cutoff
        all_ligand_index_clash_with_protein = np.array([])
        clash_ligand_index = np.array([])
    else:
        radius_sum_all = radius_ligand[:, None] + radius_protein_all[None, :]
        distance_to_radius_add_cutoff = distances_all - radius_sum_all * clash_cutoff
    if distance_to_radius_add_cutoff.size > 0:
        distance_to_radius_add_cutoff_argmin = np.unravel_index(distance_to_radius_add_cutoff.argmin(), distance_to_radius_add_cutoff.shape)
        distance_to_radius_add_cutoff_min = distance_to_radius_add_cutoff[distance_to_radius_add_cutoff_argmin]
        ##* get all ligand index that clash with protein surface
        all_ligand_index_clash_with_protein = np.unique(np.where(distance_to_radius_add_cutoff <= 0)[0])
        clash_ligand_index = np.unique(mask_ligand_idxs[all_ligand_index_clash_with_protein])
        #most_clash_ligand_idx = mask_ligand_idxs[distance_to_radius_add_cutoff_argmin[0]]
        #most_clash_protein_idx = mask_protein_idxs[distance_to_radius_add_cutoff_argmin[1]]
    else:
        distance_to_radius_add_cutoff_min = np.nan
        all_ligand_index_clash_with_protein = np.array([])
        clash_ligand_index = np.array([])
    
    if distances.size > 0:
        violations[np.unravel_index(distances.argmin(), distances.shape)] = True  # add smallest distances as check
        violations[np.unravel_index(relative_distance.argmin(), relative_distance.shape)] = True
    violation_ligand, violation_protein = np.where(violations)
    reverse_ligand_idxs = mask_ligand_idxs[violation_ligand]    #* get original ligand atom indexes that are close to protein
    reverse_protein_idxs = mask_protein_idxs[violation_protein] #* get original protein atom indexes that are close to ligand
    
    
    # collect details around those violations in a dataframe
    details = pd.DataFrame()
    details["ligand_atom_id"] = reverse_ligand_idxs
    details["protein_atom_id"] = reverse_protein_idxs
    details["ligand_element"] = [atoms_ligand[i] for i in violation_ligand]
    details["protein_element"] = [atoms_protein[i] for i in violation_protein]
    details["ligand_vdw"] = [radius_ligand[i] for i in violation_ligand]
    details["protein_vdw"] = [radius_protein[i] for i in violation_protein]
    details["sum_radii"] = details["ligand_vdw"] + details["protein_vdw"]
    details["distance"] = distances[violation_ligand, violation_protein]
    details["sum_radii_scaled"] = details["sum_radii"] * radius_scale
    details["relative_distance"] = details["distance"] / details["sum_radii_scaled"]
    details["clash"] = details["relative_distance"] < clash_cutoff

    # results = {
    #     "smallest_distance": details["distance"].min(),
    #     "not_too_far_away": details["distance"].min() <= max_distance,
    #     "num_pairwise_clashes": details["clash"].sum(),
    #     "no_clashes": not details["clash"].any(),

    #     #"distance": distances[violation_ligand, violation_protein] if len(details) > 0 else [],
    #     #"ligand_pos": [coords_ligand[i] for i in reverse_ligand_idxs] if len(details) > 0 else [],
    #     #"protein_pos": [coords_protein[i] for i in reverse_protein_idxs] if len(details) > 0 else []
    #     "distance_before_cutoff": distance_to_radius_add_cutoff_min, # min > 0 -> no_clash
    #     "close_ligand_idx": np.unique(reverse_ligand_idxs),
    #     "clash_ligand_idx": clash_ligand_index
    #     #"clash_protein_idx": np.unique(reverse_protein_idxs)
    # }

    # # add most extreme values to results table
    # #i = np.argmin(details["relative_distance"]) if len(details) > 0 else None
    # #most_extreme = {"most_extreme_" + c: details.loc[i][str(c)] if i is not None else pd.NA for c in details.columns}
    # #results = {**results, **most_extreme}

    return clash_ligand_index

def get_three_atom_rings(mol):
    """
    Get the atom indices of three-membered rings in the molecule.
    Returns a list of lists, where each inner list contains the atom indices of a 3-atom ring.
    """
    three_atom_rings = []  # Initialize list for storing 3-atom ring atom indices
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) == 3:
            three_atom_rings.extend(list(ring))  # Store the atom indices of the 3-atom ring

    return three_atom_rings

def _pairwise_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def _get_radii(atoms: np.ndarray, radius_type: str) -> np.ndarray:
    if radius_type == "vdw":
        return np.array([_periodic_table.GetRvdw(a) for a in atoms])
    elif radius_type == "covalent":
        return np.array([_periodic_table.GetRcovalent(a) for a in atoms])
    else:
        raise ValueError(f"Unknown radius type {radius_type}. Valid values are 'vdw' and 'covalent'.")


"""Module to check bond lengths, bond angles, and internal clash of ligand conformations."""

from copy import deepcopy
from logging import getLogger
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem.rdmolops import SanitizeMol

def ae(val_pred: float, val_true: float) -> float:
    """Calculate absolute error."""
    return abs(val_pred - val_true)


def pe(val_pred: float, val_true: float) -> float:
    """Calculate percentage error."""
    return (val_pred - val_true) / val_true


def ape(val_pred: float, val_true: float) -> float:
    """Calculate absolute percentage error."""
    return abs(pe(val_pred, val_true))


def bae(val: float, lb: float, ub: float) -> float:
    """Calculate out of bounds absolute error."""
    if val < lb:
        return ae(val, lb)
    if val > ub:
        return ae(val, ub)
    return 0.0

def bpe(val: float, lb: float, ub: float) -> float:
    """Calculate out of bounds percentage error."""
    if val < lb:
        return pe(val, lb)
    if val > ub:
        return pe(val, ub)
    return 0.0


def bape(val: float, lb: float, ub: float) -> float:
    """Calculate out of bounds absolute percentage error."""
    if val < lb:
        return ape(val, lb)
    if val > ub:
        return ape(val, ub)
    return 0.0


logger = getLogger(__name__)

col_lb = "lower_bound"
col_ub = "upper_bound"
col_pe = "percent_error"
col_bpe = "bound_percent_error"
col_bape = "bound_absolute_percent_error"

bound_matrix_params = {
    "set15bounds": True,
    "scaleVDW": True,
    "doTriangleSmoothing": True,
    "useMacrocycle14config": False,
}

col_n_bonds = "number_bonds"
col_shortest_bond = "shortest_bond_relative_length"
col_longest_bond = "longest_bond_relative_length"
col_n_short_bonds = "number_short_outlier_bonds"
col_n_long_bonds = "number_long_outlier_bonds"
col_n_good_bonds = "number_valid_bonds"
col_bonds_result = "bond_lengths_within_bounds"
col_n_angles = "number_angles"
col_extremest_angle = "most_extreme_relative_angle"
col_n_bad_angles = "number_outlier_angles"
col_n_good_angles = "number_valid_angles"
col_angles_result = "bond_angles_within_bounds"
col_n_noncov = "number_noncov_pairs"
col_closest_noncov = "shortest_noncovalent_relative_distance"
col_n_clashes = "number_clashes"
col_n_good_noncov = "number_valid_noncov_pairs"
col_clash_result = "no_internal_clash"

_empty_results = {
    col_n_bonds: np.nan,
    col_shortest_bond: np.nan,
    col_longest_bond: np.nan,
    col_n_short_bonds: np.nan,
    col_n_long_bonds: np.nan,
    col_bonds_result: np.nan,
    col_n_angles: np.nan,
    col_extremest_angle: np.nan,
    col_n_bad_angles: np.nan,
    col_angles_result: np.nan,
    col_n_noncov: np.nan,
    col_closest_noncov: np.nan,
    col_n_clashes: np.nan,
    col_clash_result: np.nan,
}


def check_geometry(  # noqa: PLR0913, PLR0915
    mol_pred: Mol,
    threshold_bad_bond_length: float = 0.2,
    threshold_clash: float = 0.2,
    threshold_bad_angle: float = 0.2,
    bound_matrix_params: dict[str, Any] = bound_matrix_params,
    ignore_hydrogens: bool = True,
    sanitize: bool = True,
) -> dict[str, Any]:
    """Use RDKit distance geometry bounds to check the geometry of a molecule.

    Args:
        mol_pred: Predicted molecule (docked ligand). Only the first conformer will be checked.
        threshold_bad_bond_length: Bond length threshold in relative percentage. 0.2 means that bonds may be up to 20%
            longer than DG bounds. Defaults to 0.2.
        threshold_clash: Threshold for how much overlap constitutes a clash. 0.2 means that the two atoms may be up to
            80% of the lower bound apart. Defaults to 0.2.
        threshold_bad_angle: Bond angle threshold in relative percentage. 0.2 means that bonds may be up to 20%
            longer than DG bounds. Defaults to 0.2.
        bound_matrix_params: Parameters passe to RDKit's GetMoleculeBoundsMatrix function.
        ignore_hydrogens: Whether to ignore hydrogens. Defaults to True.
        sanitize: Sanitize molecule before running DG module (recommended). Defaults to True.

    Returns:
        PoseBusters results dictionary.
    """
    ret_list = []
    mol = deepcopy(mol_pred)
    results = _empty_results.copy()

    if mol.GetNumConformers() == 0:
        logger.warning("Molecule does not have a conformer.")
        return {"results": results}

    if mol.GetNumAtoms() == 1:
        logger.warning(f"Molecule has only {mol.GetNumAtoms()} atoms.")
        results[col_angles_result] = True
        results[col_bonds_result] = True
        results[col_clash_result] = True
        return {"results": results}

    # sanitize to ensure DG works or manually process molecule
    try:
        if sanitize:
            flags = SanitizeMol(mol)
            assert flags == 0, f"Sanitization failed with flags {flags}"
    except Exception:
        return {"results": results}

    # get bonds and angles
    bond_set = sorted(_get_bond_atom_indices(mol))  # tuples
    angles = sorted(_get_angle_atom_indices(bond_set))  # triples
    angle_set = {(a[0], a[2]): a for a in angles}  # {tuples : triples}

    if len(bond_set) == 0:
        logger.warning("Molecule has no bonds.")

    # distance geometry bounds, lower triangle min distances, upper triangle max distances
    bounds = GetMoleculeBoundsMatrix(mol, **bound_matrix_params)

    # indices
    lower_triangle_idcs = np.tril_indices(mol.GetNumAtoms(), k=-1)
    upper_triangle_idcs = (lower_triangle_idcs[1], lower_triangle_idcs[0])

    # 1,2- distances
    df_12 = pd.DataFrame()
    df_12["atom_pair"] = list(zip(*upper_triangle_idcs))  # indices have i1 < i2
    df_12["atom_types"] = [
        "--".join(tuple(mol.GetAtomWithIdx(int(j)).GetSymbol() for j in i)) for i in df_12["atom_pair"]
    ]
    df_12["angle"] = df_12["atom_pair"].apply(lambda x: angle_set.get(x, None))
    df_12["has_hydrogen"] = [_has_hydrogen(mol, i) for i in df_12["atom_pair"]]
    df_12["is_bond"] = [i in bond_set for i in df_12["atom_pair"]]
    df_12["is_angle"] = df_12["angle"].apply(lambda x: x is not None)
    df_12[col_lb] = bounds[lower_triangle_idcs]
    df_12[col_ub] = bounds[upper_triangle_idcs]

    # add observed dimensions
    conformer = mol.GetConformer()
    conf_distances = _pairwise_distance1(conformer.GetPositions())
    df_12["conf_id"] = conformer.GetId()
    df_12["distance"] = conf_distances[lower_triangle_idcs]

    if ignore_hydrogens:
        df_12 = df_12.loc[~df_12["has_hydrogen"], :]

    # calculate violations
    df_bonds = _bond_check(df_12)
    df_clash = _clash_check(df_12)
    df_angles = _angle_check(df_12)

    # bond statistics
    results[col_n_bonds] = len(df_bonds)
    # print(df_bonds[col_pe] < -threshold_bad_bond_length)
    ret_list.extend(df_bonds.loc[df_bonds[col_pe] < -threshold_bad_bond_length, 'atom_pair'].tolist())
    ret_list.extend(df_bonds.loc[df_bonds[col_pe] > threshold_bad_bond_length, 'atom_pair'].tolist())
    ret_list.extend(df_angles.loc[df_angles[col_bape] > threshold_bad_angle, 'atom_pair'].tolist())
    ret_list.extend(df_clash.loc[df_clash[col_bpe] < -threshold_clash, 'atom_pair'].tolist())
    out_list = []
    for i in ret_list:
        out_list.append(i[0])
        out_list.append(i[1])
    out_list = list(set(out_list))

    return out_list


def _bond_check(df: pd.DataFrame) -> pd.DataFrame:
    # bonds can be too short or too long
    df = df[df["is_bond"]].copy()
    df[col_pe] = df.apply(lambda x: bpe(*x[["distance", col_lb, col_ub]]), axis=1)
    return df


def _angle_check(df: pd.DataFrame) -> pd.DataFrame:
    # angles have no direction (we do not know if larger or bigger beyond bounds)
    df = df[(~df["is_bond"]) & (df["is_angle"])].copy()
    df[col_bape] = df.apply(lambda x: bape(*x[["distance", col_lb, col_ub]]), axis=1)
    return df


def _clash_check(df: pd.DataFrame) -> pd.DataFrame:
    # clash is only when lower bound is violated
    df = df[(~df["is_bond"]) & (~df["is_angle"])].copy()

    def _lb_pe(value, lower_bound):
        if value >= lower_bound:
            return 0.0
        return pe(value, lower_bound)

    df[col_bpe] = df.apply(lambda x: _lb_pe(*x[["distance", col_lb]]), axis=1)
    return df


def _get_bond_atom_indices(mol: Mol) -> list[tuple[int, int]]:
    bonds = []
    for bond in mol.GetBonds():
        bond_tuple = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_tuple = _sort_bond(bond_tuple)
        bonds.append(bond_tuple)
    return bonds


def _get_angle_atom_indices(bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Check all combinations of bonds to generate list of molecule angles."""
    angles = []
    bonds = list(bonds)
    for i in range(len(bonds)):
        for j in range(i + 1, len(bonds)):
            angle = _two_bonds_to_angle(bonds[i], bonds[j])
            if angle is not None:
                angles.append(angle)
    return angles


def _two_bonds_to_angle(bond1: tuple[int, int], bond2: tuple[int, int]) -> None | tuple[int, int, int]:
    set1 = set(bond1)
    set2 = set(bond2)
    all_atoms = set1 | set2
    # angle requires two bonds to share exactly one atom, that is we must have 3 atoms
    if len(all_atoms) != 3:  # noqa: PLR2004
        return None
    # find shared atom
    shared_atom = set1 & set2
    other_atoms = all_atoms - shared_atom
    return (min(other_atoms), shared_atom.pop(), max(other_atoms))


def _sort_bond(bond: tuple[int, int]) -> tuple[int, int]:
    return (min(bond), max(bond))


def _pairwise_distance1(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def _has_hydrogen(mol: Mol, idcs: Iterable[int]) -> bool:
    return any(_is_hydrogen(mol, idx) for idx in idcs)


def _is_hydrogen(mol: Mol, idx: int) -> bool:
    return mol.GetAtomWithIdx(int(idx)).GetAtomicNum() == 1

def x_list_to_2_hop(mol_pred, x_list):
    """
    Returns the 2-hop neighbors for the atom indices listed in x_list.

    Parameters:
    mol_pred (rdkit.Chem.Mol): The molecule object from RDKit.
    x_list (list of int): A list of atom indices.

    Returns:
    list of int: A list of unique 2-hop neighbor atom indices, including 1-hop neighbors and the original atoms.
    """
    neighbor_indices = set(x_list)  # Include original atom indices
    atom_num = mol_pred.GetNumAtoms()

    # Find 1-hop neighbors
    one_hop_neighbors = set()
    for atom_index in x_list:
        atom = mol_pred.GetAtomWithIdx(int(atom_index))
        for neighbor in atom.GetNeighbors():
            one_hop_neighbors.add(neighbor.GetIdx())
    
    neighbor_indices.update(one_hop_neighbors)  # Add 1-hop neighbors

    # Find 2-hop neighbors
    for atom_index in one_hop_neighbors:
        atom = mol_pred.GetAtomWithIdx(atom_index)
        for neighbor in atom.GetNeighbors():
            neighbor_indices.add(neighbor.GetIdx())
    for atom_index in x_list:
        neighbor_indices.add(int(atom_index))
    # Convert the set to a sorted list and return
    return sorted(neighbor_indices), atom_num

def x_list_to_1_hop(mol_pred, x_list):
    """
    Returns the 1-hop neighbors for the atom indices listed in x_list.

    Parameters:
    mol_pred (rdkit.Chem.Mol): The molecule object from RDKit.
    x_list (list of int): A list of atom indices.

    Returns:
    list of int: A list of unique 1-hop neighbor atom indices.
    """
    neighbor_indices = set()
    atom_num = mol_pred.GetNumAtoms()
    # Iterate over each atom index in x_list
    for atom_index in x_list:
        # print(atom_index, mol_pred)
        # atom = mol_pred.GetAtomWithIdx(0)
        # print(atom)
        atom = mol_pred.GetAtomWithIdx(int(atom_index))
        # Iterate over neighbors of the current atom
        for neighbor in atom.GetNeighbors():
            neighbor_indices.add(neighbor.GetIdx())
        neighbor_indices.add(int(atom_index))

    # Convert the set to a sorted list and return
    return sorted(neighbor_indices), atom_num

def x_list_to_0_hop(mol_pred, x_list):
    
    atom_num = mol_pred.GetNumAtoms()

    return atom_num


import os, sys, time
current_year = time.localtime().tm_year
bad_property = []
init_dir = sys.argv[1]
os.makedirs(f'{sys.argv[1][:-1]}_fixed', exist_ok=True)
dealed_fn_list = [i.split(f'_{current_year}_')[0] for i in os.listdir(f'{sys.argv[1][:-1]}_fixed')]
for fn in os.listdir(f'{init_dir}'):
    if '_SDF' in fn and sys.argv[2] in fn:
        for sub_fns in os.listdir(f'{init_dir}/{fn}'):
            if sub_fns.endswith('pdb'):
                mol_cond = read_pdb_file(f"{init_dir}/{fn}/{sub_fns}")
                for sub_fns1 in os.listdir(f'{init_dir}/{fn}'):
                    x_list = []
                    y_list = []
                    log_path_name = fn[:-4].replace('_poc','') + sub_fns1.split(sub_fns[:-15])[-1][:-4]
                    if sub_fns1.endswith('sdf') and log_path_name not in dealed_fn_list:
                        mol_pred = read_sdf_file(f"{init_dir}/{fn}/{sub_fns1}")
                        bad_bond_angle = check_geometry(mol_pred)
                        if len(check_intermolecular_distance(mol_pred, mol_cond)) > 1:
                            x_list.extend(check_intermolecular_distance(mol_pred, mol_cond))
                        if len(get_three_atom_rings(mol_pred)) > 0:
                            x_list.extend(get_three_atom_rings(mol_pred))
                        if len(bad_bond_angle) > 0:
                            x_list.extend(bad_bond_angle)
                        

                        print(fn[:-4].replace('_poc',''), sub_fns[:-15], sub_fns1.split(sub_fns[:-15])[-1][:-4], log_path_name not in dealed_fn_list)
                        # exit()
                        
                        if len(x_list) > 0:
                            atom_num = x_list_to_0_hop(mol_pred, x_list)
                            # x_list.extend(ext_list)
                            ext_set = set(x_list)
                            # print(ext_set)
                            bad_property.append(len(ext_set)/atom_num)

                            # print(x_list, x_list_to_1_hop(mol_pred, x_list))
                            x_list = list(map(str, x_list))
                            x_str = ','.join(x_list)

                            
                            os.system(f'python sample_for_pdb_fixgen.py --pdb_path {init_dir}/{fn}/{sub_fns} --log_path_name  {log_path_name} --to_be_removed {x_str} --ligand_path {init_dir}/{fn}/{sub_fns1} --outdir {sys.argv[1][:-1]}_fixed')
                    # os.system(f'python sample_for_pdb_random.py --pdb_path {init_dir}/{fn}/{sub_fns}  --ligand_path {init_dir}/{fn}/{sub_fns1}')
print(np.mean(bad_property), np.std(bad_property))
