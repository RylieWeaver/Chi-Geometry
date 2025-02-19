# # Load Data
# train_dataset = torch.load(os.path.join(f"{datadir}/train.pt"))
# val_dataset = torch.load(os.path.join(f"{datadir}/val.pt"))
# test_dataset = torch.load(os.path.join(f"{datadir}/test.pt"))
# print(f"Datasets Loaded with Train: {len(train_dataset)}, "
#     f"Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")

# # Define a helper function to keep only the top-10k largest molecules
# def keep_largest(dataset, num_samples=10000):
#     # Sort by number of nodes (descending), then slice the first num_samples
#     dataset_sorted = sorted(dataset, key=lambda data: data.num_nodes, reverse=True)
#     return dataset_sorted[:num_samples]

# # Apply the filter to each dataset
# train_dataset = keep_largest(train_dataset, 8000)
# val_dataset = keep_largest(val_dataset, 1000)
# test_dataset = keep_largest(test_dataset, 1000)

# # Save the Dataset
# torch.save(train_dataset, os.path.join(f"{datadir}/train_largest.pt"))
# torch.save(val_dataset, os.path.join(f"{datadir}/val_largest.pt"))
# torch.save(test_dataset, os.path.join(f"{datadir}/test_largest.pt"))


# Changes
## Long data.chirality tensors not float in dataset creation (for Cross-Entropy loss)