from Bio.PDB import PDBParser
import numpy as np
import gzip
import pickle
import os

def get_atom_coordinates(residue, atom_name="CA"):
    """
    Extract the coordinates of a specific atom from a residue.
    Defaults to the alpha carbon (CA) atom.
    """
    if atom_name in residue:
        return residue[atom_name].coord
    else:
        raise ValueError(f"Atom '{atom_name}' not found in residue {residue.get_id()}.")


def calculate_distance(coord1, coord2):
    """Calculate the Euclidean distance between two 3D coordinates."""
    return np.linalg.norm(coord1 - coord2)
def find_residue_by_id(chain, res_id):
    """
    Find a residue in a chain by its residue number (ignoring heteroatom flag and insertion code).
    """
    for residue in chain.get_residues():
        # The residue ID is a tuple: (hetero flag, residue number, insertion code)
        if residue.id[1] == res_id:
            return residue
    raise ValueError(f"Residue with ID {res_id} not found in chain {chain.id}.")

def find_distances(pdb_file, target_residue_id, other_residues_ids, chain_id="A"):
    """
    Calculate the distance between a target residue and a set of other residues.
    - pdb_file: Path to the PDB file.
    - target_residue_id: ID (number) of the target residue.
    - other_residues_ids: List of residue IDs to compare with.
    - chain_id: The chain to search for residues (default: 'A').
    """
    # Parse the PDB file
    if pdb_file.endswith(".gz"):
        with gzip.open(pdb_file, "rt") as f:
            parser = PDBParser()
            structure = parser.get_structure("protein", f)
    else:
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_file)
    
    # Extract the specified chain
    chain = structure[0][chain_id]
    # Get coordinates of the target residue's CA atom
    try:
        target_residue = find_residue_by_id(chain, target_residue_id)
    except Exception as e:
        return ('{}: {}'.format(prot, e))
        
    target_coord = get_atom_coordinates(target_residue)

    # Iterate through the other residues and calculate distances
    distances = []
    for residue_id in other_residues_ids:
        try:
            other_residue = find_residue_by_id(chain, residue_id)
        except Exception as e:
            return ('{}: {}'.format(prot, e))
        other_coord = get_atom_coordinates(other_residue)
        dist = calculate_distance(target_coord, other_coord)
        distances.append(dist)

    return distances

with open('../data/protein_highAttnSite.pkl', 'rb') as file:
    prot_highAttend = pickle.load(file)
with open('../data/prot_activeSite.pkl', 'rb') as file:
    prot_activeSite = pickle.load(file)  

pdb_dir = '../human_proteome' ## directory to pdb files
dist_spatial = []

num_prot_anal = 0
for prot in prot_highAttend:

    if (prot in prot_activeSite):
        HA_sites = [x+1 for x in prot_highAttend[prot]]
        
        active_sites = prot_activeSite[prot]
        pdb_file = '{}/{}.pdb.gz'.format(pdb_dir, prot)
        if (os.path.exists(pdb_file)):
            num_prot_anal = num_prot_anal + 1
            for HA in HA_sites:
                distances = find_distances(pdb_file, HA, active_sites)
                min_dist = (min(distances))

                if (not isinstance(min_dist, str)):
                    dist_spatial.append(min_dist)

print ('Number of prot w HA, active, and PDB: {}'.format(num_prot_anal))
with open('../data/spatial_HA_active_dist.pkl', 'wb') as file:
    pickle.dump(dist_spatial,file)
