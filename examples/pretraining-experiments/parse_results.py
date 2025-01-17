#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
import numpy as np


def parse_directory_name(dir_name):
    """
    Parse directory name to identify:
      - target (combined as 'binding')
      - is_pretrained (bool)
      - number of fine-tune samples (int)
    Example directory names:
      binding_affinity_100-fine-tune-samples_2025-01-15_17-55-56
      binding_affinity_from_pretrained_100-fine-tune-samples_2025-01-15_17-45-08
      binding_efficiency_from_pretrained_1000-fine-tune-samples_2025-01-14_16-20-25
    """
    # Treat both 'binding_affinity' and 'binding_efficiency' as 'binding'
    if dir_name.startswith("binding_affinity") or dir_name.startswith(
        "binding_efficiency"
    ):
        target = "binding"  # Combine into one category
    else:
        return None

    # Check if it is pretrained
    is_pretrained = "from_pretrained" in dir_name

    # Extract the number of fine-tune samples
    # (look for "_100-fine-tune-samples_" or similar)
    match = re.search(r"_([0-9]+)-fine-tune-samples_", dir_name)
    if not match:
        return None
    num_samples = int(match.group(1))

    return target, is_pretrained, num_samples


def parse_final_results(filepath):
    """
    Read the final_results.txt file and pull out the Test Loss.
    Expects a line like:
      Test Loss:  0.5167
    Returns a float or None if not found.
    """
    test_loss = None
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("Test Loss:"):
                parts = line.split(":")
                if len(parts) == 2:
                    test_loss_str = parts[1].strip()
                    try:
                        test_loss = float(test_loss_str)
                    except ValueError:
                        test_loss = None
                break
    return test_loss


def main():
    # Look inside the logs folder for any directory starting with 'binding'
    candidate_dirs = glob.glob("logs/binding*")

    records = []

    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue

        base_dir = os.path.basename(d)  # e.g. 'binding_affinity_100-...'
        parsed = parse_directory_name(base_dir)
        if parsed is None:
            continue
        target, is_pretrained, num_samples = parsed

        final_path = os.path.join(d, "final_results.txt")
        if not os.path.isfile(final_path):
            continue

        test_loss = parse_final_results(final_path)
        if test_loss is None:
            continue

        records.append(
            {
                "target": target,
                "is_pretrained": is_pretrained,
                "num_samples": num_samples,
                "test_loss": test_loss,
            }
        )

    # Build a dataframe of all runs
    df = pd.DataFrame(records)

    if df.empty:
        print("No matching runs found or no valid 'final_results.txt' files.")
        return

    # Group by (target, is_pretrained, num_samples) and compute mean/std
    grouped = df.groupby(["target", "is_pretrained", "num_samples"])["test_loss"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()

    # Compute mean ± 2*std
    summary["lower"] = summary["mean"] - 2 * summary["std"]
    summary["upper"] = summary["mean"] + 2 * summary["std"]

    # Print the final summary
    print(summary)

    # Optionally, save to CSV
    # summary.to_csv("results_summary.csv", index=False)


if __name__ == "__main__":
    main()
