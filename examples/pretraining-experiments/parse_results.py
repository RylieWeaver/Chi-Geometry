import os
import re
import glob
import pandas as pd
import numpy as np


def parse_directory_name(dir_name):
    """
    Example dir names:
      binding_affinity-from-pretrained_local_e3nn-316-samples-repetition-1
      binding_affinity-from-scratch_local_e3nn-100-samples-repetition-2
    Returns (target, is_pretrained, num_samples, repetition).
    """
    if not dir_name.startswith("binding_affinity-"):
        return None

    target = "binding"
    is_pretrained = "from-pretrained_local_e3nn" in dir_name

    # Extract the number of samples
    match_samples = re.search(r"-(\d+)-samples-repetition-", dir_name)
    if not match_samples:
        return None
    num_samples = int(match_samples.group(1))

    # Extract the repetition
    match_rep = re.search(r"-repetition-(\d+)", dir_name)
    if not match_rep:
        return None
    repetition = int(match_rep.group(1))

    return target, is_pretrained, num_samples, repetition


def parse_final_results(filepath):
    """Reads final_results.txt, looks for 'Test Loss: X.XXXX'. Returns float or None."""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Test Loss:"):
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        return None
    return None


def one_sample_bootstrap_ci(data, ci=90, n_boot=100_000):
    """
    Standard bootstrap for a single sample (independent).
    Returns (mean_obs, lower_ci, upper_ci).
    """
    data = np.array(data, dtype=float)
    N = len(data)
    if N < 2:
        # Not enough data to bootstrap
        mean_obs = data.mean() if N else np.nan
        return mean_obs, mean_obs, mean_obs

    mean_obs = data.mean()
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = np.random.choice(data, size=N, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return mean_obs, lower, upper


def paired_bootstrap_improvement(data_pairs, n_boot=100_000, ci=95):
    """
    data_pairs: list of (pre_loss, scr_loss) from the same runs (same dataset slice).
    Returns dict with:
      mean_pretrained, mean_scratch
      diff_mean (scratch - pretrained)
      diff_ci_lower, diff_ci_upper (95% CI for diff)
      diff_pct_mean, diff_pct_ci_lower, diff_pct_ci_upper
      p_value_diff_leq_0
    """
    data_pairs = np.array(data_pairs, dtype=float)
    N = len(data_pairs)
    if N == 0:
        return {
            "mean_pretrained": np.nan,
            "mean_scratch": np.nan,
            "diff_mean": np.nan,
            "diff_ci_lower_95": np.nan,
            "diff_ci_upper_95": np.nan,
            "diff_pct_mean": np.nan,
            "diff_pct_ci_lower_95": np.nan,
            "diff_pct_ci_upper_95": np.nan,
            "p_value_diff_leq_0": np.nan,
        }

    # Observed
    pre_vals = data_pairs[:, 0]
    scr_vals = data_pairs[:, 1]
    mean_pre = pre_vals.mean()
    mean_scr = scr_vals.mean()

    scr_minus_pre = scr_vals - pre_vals
    diff_mean_obs = scr_minus_pre.mean()
    diff_pct_obs = diff_mean_obs / mean_scr if mean_scr != 0 else np.nan

    if N < 2:
        # With only 1 pair, no real bootstrap distribution:
        return {
            "mean_pretrained": mean_pre,
            "mean_scratch": mean_scr,
            "diff_mean": diff_mean_obs,
            "diff_ci_lower_95": diff_mean_obs,
            "diff_ci_upper_95": diff_mean_obs,
            "diff_pct_mean": diff_pct_obs,
            "diff_pct_ci_lower_95": diff_pct_obs,
            "diff_pct_ci_upper_95": diff_pct_obs,
            "p_value_diff_leq_0": float(diff_mean_obs <= 0),
        }

    # Paired bootstrap
    diffs = np.empty(n_boot, dtype=float)
    diffs_pct = np.empty(n_boot, dtype=float)
    count_diff_leq_0 = 0

    for i in range(n_boot):
        idx = np.random.randint(0, N, size=N)
        sample_pairs = data_pairs[idx]
        # compute (scratch - pretrained)
        diff_i = (sample_pairs[:, 1] - sample_pairs[:, 0]).mean()
        diffs[i] = diff_i

        mean_scr_i = sample_pairs[:, 1].mean()
        diffs_pct[i] = diff_i / mean_scr_i if mean_scr_i != 0 else np.nan

        if diff_i <= 0:
            count_diff_leq_0 += 1

    alpha = (100 - ci) / 2
    diff_lower = np.percentile(diffs, alpha)
    diff_upper = np.percentile(diffs, 100 - alpha)

    diffs_pct_valid = diffs_pct[~np.isnan(diffs_pct)]
    if len(diffs_pct_valid) > 0:
        diff_pct_lower = np.percentile(diffs_pct_valid, alpha)
        diff_pct_upper = np.percentile(diffs_pct_valid, 100 - alpha)
    else:
        diff_pct_lower = np.nan
        diff_pct_upper = np.nan

    p_value = count_diff_leq_0 / n_boot

    return {
        "mean_pretrained": mean_pre,
        "mean_scratch": mean_scr,
        "diff_mean": diff_mean_obs,  # scratch - pretrained
        "diff_ci_lower_95": diff_lower,
        "diff_ci_upper_95": diff_upper,
        "diff_pct_mean": diff_pct_obs,
        "diff_pct_ci_lower_95": diff_pct_lower,
        "diff_pct_ci_upper_95": diff_pct_upper,
        "p_value_diff_leq_0": p_value,
    }


def main():
    candidate_dirs = glob.glob("logs_full-trained2/binding_affinity-*")
    records = []

    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue

        base_dir = os.path.basename(d)
        parsed = parse_directory_name(base_dir)
        if parsed is None:
            continue

        target, is_pretrained, num_samples, repetition = parsed
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
                "repetition": repetition,
                "test_loss": test_loss,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        print("No matching runs found or no valid 'final_results.txt' files.")
        return

    # Separate pretrained vs scratch
    df_pre = df[df["is_pretrained"]].rename(columns={"test_loss": "pre_loss"})
    df_scr = df[~df["is_pretrained"]].rename(columns={"test_loss": "scr_loss"})

    # Merge on (target, num_samples, repetition) => get pairs
    df_merged = pd.merge(
        df_pre, df_scr, on=["target", "num_samples", "repetition"], how="inner"
    )

    # Create list of (pre_loss, scr_loss)
    df_merged["pair"] = list(zip(df_merged["pre_loss"], df_merged["scr_loss"]))

    # Group by (target, num_samples) => gather all pairs
    grouped_pairs = (
        df_merged.groupby(["target", "num_samples"])["pair"].apply(list).reset_index()
    )

    results = []
    for _, row in grouped_pairs.iterrows():
        tgt = row["target"]
        ns = row["num_samples"]
        pairs = row["pair"]  # list of (pre_loss, scr_loss)

        # (A) Paired difference stats, 95% CI
        diff_stats = paired_bootstrap_improvement(pairs, n_boot=100_000, ci=95)

        # (B) Single-sample bootstrap for each model (90% CI)
        pre_losses = [p[0] for p in pairs]
        scr_losses = [p[1] for p in pairs]

        _, pre_lower_90, pre_upper_90 = one_sample_bootstrap_ci(
            pre_losses, ci=90, n_boot=100_000
        )
        _, scr_lower_90, scr_upper_90 = one_sample_bootstrap_ci(
            scr_losses, ci=90, n_boot=100_000
        )

        results.append(
            {
                "target": tgt,
                "num_samples": ns,
                "n_pairs": len(pairs),
                # Means (both 95% difference + single-sample share the same means)
                "mean_pretrained": diff_stats["mean_pretrained"],
                "mean_scratch": diff_stats["mean_scratch"],
                # 90% CIs for each model alone
                "pretrained_ci90_lower": pre_lower_90,
                "pretrained_ci90_upper": pre_upper_90,
                "scratch_ci90_lower": scr_lower_90,
                "scratch_ci90_upper": scr_upper_90,
                # Paired difference (95% CI)
                "diff_mean": diff_stats["diff_mean"],
                "diff_ci_lower_95": diff_stats["diff_ci_lower_95"],
                "diff_ci_upper_95": diff_stats["diff_ci_upper_95"],
                # Percent difference relative to scratch
                "diff_pct_mean": diff_stats["diff_pct_mean"],
                "diff_pct_ci_lower_95": diff_stats["diff_pct_ci_lower_95"],
                "diff_pct_ci_upper_95": diff_stats["diff_pct_ci_upper_95"],
                "p_value_diff_leq_0": diff_stats["p_value_diff_leq_0"],
            }
        )

    out_df = pd.DataFrame(results).sort_values(["target", "num_samples"])
    print(out_df.to_string(index=False))

    # Optionally save:
    out_df.to_csv("results_improvement.csv", index=False)


if __name__ == "__main__":
    main()
