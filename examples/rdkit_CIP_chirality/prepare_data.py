# General
import pandas as pd
import logging
from tqdm import tqdm

# Torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

cip_map = {"N/A": 0, "R": 1, "S": 2}


def smiles_to_rdkit(
    smiles,
    max_number_of_atoms: int = 250,
    maxAttempts: int = 500,
    maxIters: int = 1000,
    force_field: str = "UFF",
):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Invalid SMILES string: {smiles}. Skipping molecule.")
            return None

        # Canonicalize SMILES and handle multiple fragments (e.g., salts)
        smiles_canonical = Chem.MolToSmiles(mol, canonical=True)
        primary_smiles = smiles_canonical.split(".")[0]
        mol = Chem.MolFromSmiles(primary_smiles)
        if mol is None:
            logging.warning(
                f"Failed to parse primary fragment from SMILES: {smiles}. Skipping molecule."
            )
            return None

        atom_count = mol.GetNumAtoms()
        if atom_count > max_number_of_atoms:
            logging.warning(
                f"Omitting molecule {primary_smiles} as it contains {atom_count} atoms (limit: {max_number_of_atoms})."
            )
            return None
        if atom_count == 0:
            logging.warning(
                f"Omitting molecule {primary_smiles} as it contains no atoms after processing."
            )
            return None

        mol = Chem.AddHs(mol)

        # Embed molecule with specified attempts
        embed_status = AllChem.EmbedMolecule(mol, maxAttempts=maxAttempts, randomSeed=0)
        if embed_status < 0:
            logging.info(
                f"Initial embedding failed for molecule {primary_smiles}. Retrying with random coordinates."
            )
            embed_status = AllChem.EmbedMolecule(
                mol, useRandomCoords=True, maxAttempts=maxAttempts, randomSeed=0
            )
        if embed_status < 0:
            logging.warning(
                f"Embedding failed for molecule {primary_smiles}. Skipping molecule."
            )
            return None

        # Optimize geometry with specified force field
        try:
            if force_field.upper() == "UFF":
                optimize_status = AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
            elif force_field.upper() == "MMFF":
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if mmff_props is None:
                    logging.warning(
                        f"MMFF properties not found for molecule {primary_smiles}. Falling back to UFF."
                    )
                    optimize_status = AllChem.UFFOptimizeMolecule(
                        mol, maxIters=maxIters
                    )
                else:
                    optimize_status = AllChem.MMFFOptimizeMolecule(
                        mol, mmff_props, maxIters=maxIters
                    )
            else:
                logging.warning(
                    f"Unsupported force field: {force_field}. Using UFF by default."
                )
                optimize_status = AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)

            if optimize_status != 0:
                logging.warning(
                    f"Optimization did not converge properly for molecule {primary_smiles}. Skipping molecule."
                )
                return None

        except Exception as e:
            logging.warning(
                f"Optimization failed for molecule {primary_smiles} with error: {e}. Skipping molecule."
            )
            return None

        # Assign stereochemistry
        try:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        except Exception as e:
            logging.warning(
                f"Failed to assign stereochemistry for molecule {primary_smiles} with error: {e}. Skipping molecule."
            )
            return None

        return mol

    except Exception as e:
        logging.error(
            f"An unexpected error occurred while processing SMILES {['ID']}: {e}. Skipping molecule."
        )
        return None


def rdkit_mol_to_pyg(mol):
    # Ensure stereochemistry is assigned so that CIP labels, chiral tags, etc. are set
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    # Prepare lists for node data
    atomic_numbers = []
    rdkit_tag = []
    cip_tag = []
    cip_strs = []
    positions = []

    # Get the 3D conformer
    conf = mol.GetConformer()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        # Atomic number
        atomic_numbers.append(atom.GetAtomicNum())

        # Local chiral tag (RDKit enum: CHI_UNSPECIFIED=0, CHI_TETRAHEDRAL_CW=1, etc.)
        rdkit_tag.append(int(atom.GetChiralTag()))

        # CIP label (if assigned, this is typically 'R' or 'S')
        if atom.HasProp("_CIPCode"):
            cip_str = atom.GetProp("_CIPCode")
        else:
            cip_str = "N/A"
        cip_tag.append(cip_map[cip_str])
        cip_strs.append(cip_str)

        # 3D coordinates
        pos = conf.GetAtomPosition(idx)
        positions.append([pos.x, pos.y, pos.z])

    # Build edge_index from the bonds in the molecule
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    # Assign tensors
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.float32).view(-1, 1)
    atomic_numbers_one_hot = F.one_hot(
        atomic_numbers.long() - 1, num_classes=118
    ).float()  # Subtract 1 because atomic numbers start from 1
    rdkit_tag = torch.tensor(rdkit_tag, dtype=torch.long)
    cip_tag = torch.tensor(cip_tag, dtype=torch.long).view(-1, 1)
    cip_onehot = F.one_hot(cip_tag, num_classes=3).float()
    positions = torch.tensor(positions, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create a PyG Data object
    data = Data(
        x=atomic_numbers_one_hot.float(),
        y=cip_tag,
        z=atomic_numbers.float(),
        pos=positions,
        atomic_numbers=atomic_numbers,
        atomic_numbers_one_hot=atomic_numbers_one_hot,
        rdkit_tag=rdkit_tag,
        cip_tag=cip_tag,
        cip_onehot=cip_onehot,
        cip_strs=cip_strs,
    )

    # # Debugging
    # ## Check for two chiral centers
    # sorted_tags, sorted_indices = torch.sort(data.cip_tag, descending=True)
    # if sorted_tags[1] > 0:
    #     print("Found two chiral centers")
    # ## Check for difference between CIP and RdKit tag
    # mismatch_mask = (data.rdkit_tag != data.cip_tag)
    # if mismatch_mask.any():
    #     print("Some atoms have different RDKit chiral tags vs CIP labels.")

    return data


def main():
    samples_map = {"train": 8000, "val": 1000, "test": 1000}

    # Process the raw datasets
    datadir = "datasets"
    for split in ["train", "val", "test"]:
        raw_data = pd.read_pickle(f"{datadir}/{split}_raw.pkl")
        molecules = []
        for idx, row in tqdm(
            raw_data.iterrows(),
            desc=f"Processing {split} molecules",
            total=len(raw_data),
        ):
            # Go from smiles --> rdkit --> pyg
            mol = smiles_to_rdkit(row["ID"])
            if mol is not None:
                mol = rdkit_mol_to_pyg(mol)
                molecules.append(mol)

        # Save as a PyG dataset and a testing subset
        torch.save(molecules, f"{datadir}/{split}_processed.pt")
        num_samples = samples_map[split]
        torch.save(molecules[:num_samples], f"{datadir}/{split}_processed_sample.pt")


if __name__ == "__main__":
    main()
