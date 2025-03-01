# General
import os
import json
import pickle
import requests
from tqdm import tqdm

# Torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# RdKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Chi-Geometry
from examples.utils import (
    get_avg_degree,
    get_avg_nodes,
)


# Define dataset download URLs
urls = {
    "train": "https://figshare.com/ndownloader/files/30975697?private_link=e23be65a884ce7fc8543",
    "val": "https://figshare.com/ndownloader/files/30975706?private_link=e23be65a884ce7fc8543",
    "test": "https://figshare.com/ndownloader/files/30975682?private_link=e23be65a884ce7fc8543",
}


# Function to download a file
def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as file:
        for data in tqdm(
            response.iter_content(1024), total=total_size // 1024, unit="KB"
        ):
            file.write(data)


# Function to convert RDKit molecule to PyTorch Geometric Data object
def mol_to_pyg_data(row, mol):
    atom_features = []
    atom_y = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atomic_one_hot = F.one_hot(
            torch.tensor(atomic_number - 1), num_classes=118
        ).float()
        atom_features.append(atomic_one_hot)
    atom_features = torch.stack(atom_features)

    # Edge index and edge features
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()  # 1.0 for SINGLE, 2.0 for DOUBLE, etc.
        is_aromatic = int(bond.GetIsAromatic())  # 1 if aromatic, 0 otherwise
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected graph
        edge_features.append([bond_type, is_aromatic])
        edge_features.append([bond_type, is_aromatic])  # Reverse edge
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # 3D positions
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        positions = []
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            positions.append([pos.x, pos.y, pos.z])
        pos = torch.tensor(positions, dtype=torch.float)
    else:
        raise ValueError(f'Molecule {row["ID"]} does not have 3D coordinates')

    # Target value
    binding_affinity = torch.tensor(row["top_score"], dtype=torch.float).unsqueeze(0)

    # Create PyTorch Geometric Data object
    data = Data(
        x=atom_features,
        atomic_numbers_one_hot=atom_features,
        y=binding_affinity,
        edge_index=edge_index,
        edge_attr=edge_features,
        pos=pos,
    )

    return data


# Function to process the downloaded pickle files and convert to PyTorch Geometric dataset
def process(df, split):
    data_list = []
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Processing {split} dataset"
    ):
        smiles = row["ID"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                mol = Chem.AddHs(mol)  # Add explicit hydrogens
                AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
                AllChem.UFFOptimizeMolecule(mol)  # Optimize using UFF
                pyg_data = mol_to_pyg_data(row, mol)
                data_list.append(pyg_data)
            except Exception as e:
                print(f"Skipping molecule {smiles} due to error: {e}")

    return data_list


# Define a helper function to keep only largest molecules
def keep_largest(dataset, num_samples=10000):
    # Sort by number of nodes (descending), then slice the first num_samples
    dataset_sorted = sorted(dataset, key=lambda data: data.num_nodes, reverse=True)
    return dataset_sorted[:num_samples]


def main():
    # Arguments
    datadir = "./datasets"
    os.makedirs(datadir, exist_ok=True)
    num_samples_map = {"train": 8000, "val": 1000, "test": 1000}

    # Download datasets
    for split, url in urls.items():
        download_file(url, f"{datadir}/{split}.pickle")

    # Process and save datasets
    for split in ["train", "val", "test"]:
        # Load split
        pickle_file = f"{datadir}/{split}.pickle"
        with open(pickle_file, "rb") as f:
            df = pickle.load(f)

        # Whole dataset
        # dataset = process(df, split)
        # torch.save(dataset, f"{datadir}/{split}.pt")
        dataset = torch.load(f"{datadir}/{split}.pt")

        # Sample
        sample = dataset[: num_samples_map[split]]
        torch.save(sample, f"{datadir}/{split}_sample.pt")

        # Keep graphs with the most nodes
        largest = keep_largest(dataset, num_samples_map[split])
        torch.save(largest, f"{datadir}/{split}_largest.pt")

        # Compute and save stats for each dataset
        for name, subset in [
            ("full", dataset),
            ("sample", sample),
            ("largest", largest),
        ]:
            avg_deg = get_avg_degree(subset)
            avg_nodes = get_avg_nodes(subset)
            stats = {"avg_degree": avg_deg, "avg_nodes": avg_nodes}

            # Save stats to JSON
            stats_filename = f"{split}_{name}_stats.json"  # e.g., train_full_stats.json
            stats_path = os.path.join(datadir, stats_filename)
            with open(stats_path, "w") as f_stats:
                json.dump(stats, f_stats, indent=2)


if __name__ == "__main__":
    main()
