from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem 
from rdkit import Geometry
from collections import defaultdict
import copy
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles
    
class PDBFile:
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]        
    def add(self, coords, order, part=0, repeat=1):
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}
        
    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += 'MODEL\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)
            
            
            
class ZincMolecule:
    def __init__(self, num_atoms=40):
        """
        Initialize the molecule with a specified number of zinc atoms.
        """
        self.mol = Chem.RWMol()
        self.num_atoms = num_atoms
        # Add zinc atoms
        for _ in range(num_atoms):
            self.mol.AddAtom(Chem.Atom(30))  # 30 is the atomic number for zinc

    def update_positions(self, coordinates):
        """
        Update the positions of the atoms in the molecule.
        coordinates: List of tuples with the new coordinates (x, y, z) for each atom.
        """
        if len(coordinates) != self.num_atoms:
            raise ValueError("Number of coordinates must match the number of atoms in the molecule.")
        
        conf = Chem.Conformer(self.num_atoms)
        for i, coords in enumerate(coordinates):
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*coords))
        
        # Update or add the conformer to the molecule
        if self.mol.GetNumConformers() == 0:
            self.mol.AddConformer(conf)
        else:
            self.mol.GetConformer().SetPositions([Chem.rdGeometry.Point3D(*coords) for coords in coordinates])

    def save_to_pdb(self, file_path):
        """
        Save the current state of the molecule to a PDB file.
        file_path: Path to the PDB file.
        """
        rdmolfiles.MolToPDBFile(self.mol, file_path)

    # def save_batch_to_pdb(self, coordinates_batch_tensor, file_path_template):
    #     """
    #     Save a batch of states to PDB files, each representing a different timestep, from a tensor.
    #     coordinates_batch_tensor: Tensor of shape (num_timesteps, num_atoms, 3) containing the coordinates for each timestep.
    #     file_path_template: A template for generating the file paths, which should include a placeholder for the timestep.
    #     """
    #     coordinates_batch = coordinates_batch_tensor.numpy()  # Convert to NumPy array if using PyTorch

    #     for timestep, coordinates in enumerate(coordinates_batch):
    #         # Convert the coordinates for the timestep from an array to a list of tuples
    #         coordinates_list = [tuple(coord) for coord in coordinates]
    #         self.update_positions(coordinates_list)
    #         file_path = file_path_template.format(timestep=timestep)
    #         self.save_to_pdb(file_path)

def save_zinc_to_pdb_file(zinc_coords, file_name):
    with open(file_name, "w") as file:
        for i, coords in enumerate(zinc_coords, start=1):
            # Format the line as per PDB format specifications for an ATOM record
            # This is a simplification; a real PDB file might need more detailed info
            line = f"ATOM  {i:>5}  ZN  ZN A{1:>4}    {coords[0]:>8.3f}{coords[1]:>8.3f}{coords[2]:>8.3f}  1.00 20.00          ZN\n"
            file.write(line)