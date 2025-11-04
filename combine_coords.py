import os
import shutil

move_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/CDR_filtered_coords'
original_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/CDR_filtered1_regen'
# data_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/sample_0722'

# for dirname in os.listdir(data_path):
#     dirpath = os.path.join(data_path, dirname)
#     taskname = dirname.split('_')[0]
#     for filename in os.listdir(dirpath):
#         if 'origin' in filename: 
#             filepath = os.path.join(dirpath, filename)
#             if not os.path.isfile(filepath): continue
#             new_file_name = filename.split('_origin')[0] + filename.split('_origin')[1] # this is target
#             new_antibody_name = new_file_name.replace('antigen', 'antibody')

#             if os.path.exists(os.path.join(move_path, new_file_name)): continue
#             else: 
#                 shutil.copy(filepath, os.path.join(move_path, new_file_name))
#                 print(f"Copied {filename} to {move_path}")
#             if os.path.exists(os.path.join(original_path, new_antibody_name)): continue
#             else:
#                 shutil.copy(os.path.join(original_path, new_antibody_name), os.path.join(move_path, new_antibody_name))
#                 print(f"Copied {os.path.join(original_path, new_antibody_name)} to {move_path}")

output_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/CDR_filtered_use'

import numpy as np

def read_pdb(pdb_path):
    """
    Read a PDB file and extract atom coordinates along with metadata.
    Returns a list of atom dictionaries containing all relevant information.
    
    Args:
        pdb_path (str): Path to the PDB file
        
    Returns:
        list: List of atom dictionaries with keys: 
              'line', 'name', 'residue', 'chain', 'resid', 'x', 'y', 'z'
    """
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Extract atom information from PDB formatted lines
                atom_data = {
                    'line': line.strip(),
                    'name': line[12:16].strip(),
                    'residue': line[17:20].strip(),
                    'chain': line[21],
                    'resid': int(line[22:26].strip()),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54])
                }
                atoms.append(atom_data)
    return atoms

def compute_displacement(orig_atoms, translated_atoms):
    """
    Calculate the displacement vector between two sets of atoms by comparing centroids.
    
    Args:
        orig_atoms (list): Atoms from original structure
        translated_atoms (list): Atoms from translated structure
        
    Returns:
        numpy.array: Displacement vector [dx, dy, dz]
    """
    # Calculate centroid of original structure
    orig_center = np.mean([[a['x'], a['y'], a['z']] for a in orig_atoms], axis=0)
    
    # Calculate centroid of translated structure
    trans_center = np.mean([[a['x'], a['y'], a['z']] for a in translated_atoms], axis=0)
    
    # Compute displacement vector
    displacement = trans_center - orig_center
    return displacement

def apply_displacement(atoms, displacement):
    """
    Apply a displacement vector to a set of atoms.
    
    Args:
        atoms (list): List of atom dictionaries
        displacement (numpy.array): Displacement vector [dx, dy, dz]
        
    Returns:
        list: Updated list of atoms with new coordinates
    """
    displaced_atoms = []
    for atom in atoms:
        # Create a copy of the atom dictionary
        new_atom = atom.copy()
        # Apply displacement to coordinates
        new_atom['x'] += displacement[0]
        new_atom['y'] += displacement[1]
        new_atom['z'] += displacement[2]
        displaced_atoms.append(new_atom)
    return displaced_atoms

def write_pdb(output_path, atoms):
    """
    Write atoms to a PDB file in standard format.
    
    Args:
        output_path (str): Path to output PDB file
        atoms (list): List of atom dictionaries to write
    """
    with open(output_path, 'w') as f:
        for i, atom in enumerate(atoms, start=1):
            # Format coordinates to fixed width (PDB standard)
            x, y, z = atom['x'], atom['y'], atom['z']
            coord_str = f"{x:8.3f}{y:8.3f}{z:8.3f}"
            
            # Write atom record in standard PDB format
            f.write(
                f"{atom['line'][:30]}{coord_str}{atom['line'][54:]}\n"
            )

def get_replaceed_antibody(original_antigen_pdb, original_antibody_pdb, translated_antigen_pdb, output_antigen_pdb, output_antibody_pdb):
    """
    Get the displaced antibody structure based on the original antigen and translated antigen.
    
    Args:
        original_antigen_pdb (str): Path to original antigen PDB file
        original_antibody_pdb (str): Path to original antibody PDB file
        translated_antigen_pdb (str): Path to translated antigen PDB file
        output_antibody_pdb (str): Path to save the displaced antibody PDB file
    """
    # Read antigen structures
    print(f"Reading original antigen from: {original_antigen_pdb}")
    print(f"Reading translated antigen from: {translated_antigen_pdb}")
    print(f"Reading original antibody from: {original_antibody_pdb}")
    print(f"Outputting displaced antibody to: {output_antibody_pdb}")
    print(f"Outputting translated antigen to: {output_antigen_pdb}")
    
    
    
    orig_antigen = read_pdb(original_antigen_pdb)
    trans_antigen = read_pdb(translated_antigen_pdb)
    
    # Calculate displacement vector between antigen structures
    displacement = compute_displacement(orig_antigen, trans_antigen)
    print(f"Calculated displacement vector: {displacement}")
    
    # Read original antibody structure
    original_antibody = read_pdb(original_antibody_pdb)
    
    displaced_antibody = apply_displacement(original_antibody, displacement)

    shutil.copy(translated_antigen_pdb, output_antigen_pdb)  # Copy translated antigen to output path
    # Write the displaced antibody to new PDB file
    write_pdb(output_antibody_pdb, displaced_antibody)
    print(f"Displaced antibody saved to: {output_antibody_pdb}")
   

for file in os.listdir(move_path):
    filename = file.split('_')[0]
    if 'antigen' not in file: continue
    for file2 in os.listdir(original_path):
        if filename in file2:
            if 'antigen' in file2 and '.pdb' in file2:
                original_antigen_path = os.path.join(original_path, file2)
            if 'antibody' in file2 and '.pdb' in file2:
                original_antibody_path = os.path.join(original_path, file2)
    
    print(f"Processing {file} with original antigen {original_antigen_path} and antibody {original_antibody_path}")
    
    # read file, get the coordinates
    get_replaceed_antibody(
        original_antibody_pdb= original_antibody_path,
        original_antigen_pdb= original_antigen_path,
        translated_antigen_pdb= os.path.join(move_path, file),
        output_antigen_pdb= os.path.join(output_path, file),
        output_antibody_pdb= os.path.join(output_path, file.replace('antigen', 'antibody'))
    )