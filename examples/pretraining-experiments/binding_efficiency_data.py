import os
import json
import torch
import pickle
import requests
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from hydragnn.preprocess import update_predicted_values

# Define dataset download URLs
urls = {
    "train": "https://figshare.com/ndownloader/files/30975697?private_link=e23be65a884ce7fc8543",
    "val": "https://figshare.com/ndownloader/files/30975706?private_link=e23be65a884ce7fc8543",
    "test": "https://figshare.com/ndownloader/files/30975682?private_link=e23be65a884ce7fc8543",
}

# Define folder to store downloaded data
dataset_dir = "./binding_affinity_data"
os.makedirs(dataset_dir, exist_ok=True)

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
        atomic_one_hot = F.one_hot(torch.tensor(atomic_number), num_classes=118)
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
    binding_efficiency = torch.tensor(row["top_score"], dtype=torch.float).unsqueeze(0)

    # Create PyTorch Geometric Data object
    data = Data(
        x=atom_features,
        y=binding_efficiency,
        edge_index=edge_index,
        edge_attr=edge_features,
        pos=pos,
    )

    return data


# Function to process the downloaded pickle files and convert to PyTorch Geometric dataset
def process_and_save(split):
    pickle_file = f"{dataset_dir}/{split}.pickle"
    with open(pickle_file, "rb") as f:
        data_df = pickle.load(f)

    data_list = []
    data_df = data_df.sample(frac=0.01)  # Shuffle the dataset
    # data_df = data_df[:3000]
    for idx, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc=f"Processing {split} dataset"
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

    # Save the processed dataset
    torch.save(data_list, f"{dataset_dir}/{split}.pt")


def main():
    # Download datasets
    for split, url in urls.items():
        download_file(url, f"{dataset_dir}/{split}.pickle")

    # Process and save datasets
    for split in ["train", "val", "test"]:
        process_and_save(split)


if __name__ == "__main__":
    main()
