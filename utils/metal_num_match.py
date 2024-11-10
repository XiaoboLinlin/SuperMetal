from rdkit import Chem
from rdkit.Geometry import rdGeometry

def match_num_metal(mol, num_metal):
    # Ensure the molecule is a writable molecule
    mol = Chem.RWMol(mol)

    # Get the current number of atoms
    original_atom_num = mol.GetNumAtoms()

    # Check if the current number of atoms exceeds the desired count
    if original_atom_num > num_metal:
        raise ValueError("The original number of atoms is greater than the specified number of metal atoms.")

    # Add the required number of zinc atoms to match the desired total
    for _ in range(num_metal - original_atom_num):
        mol.AddAtom(Chem.Atom("Zn"))

    # Check if there is an existing conformer and remove it
    if mol.GetNumConformers() > 0:
        for conf_id in [conf.GetId() for conf in mol.GetConformers()]:
            mol.RemoveConformer(conf_id)

    # Create a new conformer with all positions set to zero
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        # Correctly creating a Point3D object to set positions
        conf.SetAtomPosition(i, rdGeometry.Point3D(0, 0, 0))  # Set all positions to (0,0,0)

    # Add the conformer to the molecule, consider handling the ID explicitly
    conf_id = mol.AddConformer(conf, assignId=True)

    # Debug: Output the assigned conformer ID
    print(f"Assigned Conformer ID: {conf_id}")

    # Convert back to a non-writable Mol object
    return mol.GetMol()