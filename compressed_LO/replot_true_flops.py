import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# USER CONFIG
# ----------------------------------------------------------------------

CSV_PATHS = [
    "multi_dataset_flops_benchmark.csv",
]

MODEL_INFO = {
    "RNNLM_lm1b32k_Patch32_LSTM256_Embed128": {
        "n_params": 12_753_252,   # TODO: replace
        "batch_size": 32,         # TODO: replace
    },
    "RNNLM_lm1bbytes_Patch32_LSTM128_Embed64": {
        "n_params": 149_059,      # TODO: replace
        "batch_size": 32,         # TODO: replace
    },
    "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128": {
        "n_params": 12_753_252,   # TODO: replace
        "batch_size": 32,         # TODO: replace
    },
    "TransformerLM_LM1B_MultiRuntime_0": {
        "n_params": 1_332_820,    # TODO: replace
        "batch_size": 32,         # TODO: replace
    },
}

CELO_PER_PARAM_FLOPS      = 288.0
ADAM_PER_PARAM_FLOPS      = 16.0
ADAFACTOR_PER_PARAM_FLOPS = 12.0
SGD_PER_PARAM_FLOPS       = 2.0

CELO_PRUNED_PER_PARAM_FLOPS = 136.0
CELO_Q8_PER_PARAM_FLOPS     = CELO_PER_PARAM_FLOPS

TRUE_FLOPS_PLOT_PREFIX = "trueflops"

# Match your other plots: short title + default figure aspect ratio
PLOT_TITLE = "Optimizers Training Loss vs. FLOPs"

# Optional: save into a folder like your other script
OUTPUT_DIR = "./flops_experiment_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Match legend order / naming from your other script
OPT_ORDER = ["celo", "celo_sam", "adam", "sgd", "adafactor"]
OPT_LABEL = {
    "celo": "CeLO",
    "celo_sam": "CeLO-SAM",
    "adam": "Adam",
    "sgd": "SGD",
    "adafactor": "Adafactor",
}


def compute_flops_per_step(dataset_name: str, optimizer_name: str) -> float:
    """Return FLOPs per training step for a given dataset and optimizer:
       FLOPs_step = N_model * ( 6 * B + c_opt ).
    """
    if dataset_name not in MODEL_INFO:
        raise ValueError(f"No MODEL_INFO entry for dataset '{dataset_name}'")

    info = MODEL_INFO[dataset_name]
    n_model = float(info["n_params"])
    batch_size = float(info["batch_size"])

    model_flops_per_param = 6.0 * batch_size

    name = optimizer_name.lower()
    if name.startswith("celo2"):
        opt_flops_per_param = CELO_PER_PARAM_FLOPS
    elif name.startswith("celo_prune") or "celo_prune" in name:
        opt_flops_per_param = CELO_PRUNED_PER_PARAM_FLOPS
    elif name.startswith("celo_q8") or "celo_q8" in name:
        opt_flops_per_param = CELO_Q8_PER_PARAM_FLOPS
    elif name.startswith("celo"):
        opt_flops_per_param = CELO_PER_PARAM_FLOPS
    elif name.startswith("adam"):
        opt_flops_per_param = ADAM_PER_PARAM_FLOPS
    elif name.startswith("adafactor"):
        opt_flops_per_param = ADAFACTOR_PER_PARAM_FLOPS
    elif name.startswith("sgd"):
        opt_flops_per_param = SGD_PER_PARAM_FLOPS
    else:
        opt_flops_per_param = ADAM_PER_PARAM_FLOPS

    return n_model * (model_flops_per_param + opt_flops_per_param)


def make_true_flops_plots(csv_paths, plot_prefix=TRUE_FLOPS_PLOT_PREFIX):
    # Load and concatenate all CSVs
    dfs = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV path '{p}' not found")
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

    required_cols = {"dataset", "run", "step", "optimizer", "train_loss", "val_loss", "cum_flops"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns. Found columns: {df.columns.tolist()}")

    df = df.copy()
    df["true_cum_flops"] = np.nan

    # Compute true_cum_flops = step * FLOPs_step(dataset, optimizer)
    for dataset in df["dataset"].unique():
        for opt_name in df.loc[df["dataset"] == dataset, "optimizer"].unique():
            flops_step = compute_flops_per_step(dataset, opt_name)
            mask = (df["dataset"] == dataset) & (df["optimizer"] == opt_name)
            df.loc[mask, "true_cum_flops"] = df.loc[mask, "step"].to_numpy(dtype=float) * flops_step

    # Plot: one per dataset, mean ± std across runs
    for dataset in sorted(df["dataset"].unique()):
        df_d = df[df["dataset"] == dataset].copy()

        # Build deterministic optimizer iteration order
        ordered_opt_names = []
        for o_low in OPT_ORDER:
            ordered_opt_names.extend([o for o in df_d["optimizer"].unique() if o.lower() == o_low])
        remaining = [o for o in df_d["optimizer"].unique() if o not in ordered_opt_names]
        ordered_opt_names.extend(sorted(remaining, key=lambda s: s.lower()))

        # -------------------------
        # PASS 1: compute per-optimizer grouped curves and find fairness cutoff
        # -------------------------
        curves = {}  # opt_name -> (x, y, y_std)
        final_xs = []

        for opt_name in ordered_opt_names:
            sub_all = df_d[df_d["optimizer"] == opt_name].copy()
            if sub_all.empty:
                continue

            grouped = (
                sub_all
                .groupby("step", as_index=False)
                .agg(
                    mean_true_flops=("true_cum_flops", "mean"),
                    mean_train=("train_loss", "mean"),
                    std_train=("train_loss", "std"),
                )
                .sort_values("mean_true_flops")
            )
            grouped["std_train"] = grouped["std_train"].fillna(0.0)

            x = grouped["mean_true_flops"].to_numpy(dtype=float)
            y = grouped["mean_train"].to_numpy(dtype=float)
            y_std = grouped["std_train"].to_numpy(dtype=float)

            # Skip degenerate curves
            if len(x) == 0 or not np.isfinite(x).any():
                continue

            curves[opt_name] = (x, y, y_std)
            # "final x" for this curve is its last x-value (after sorting)
            final_xs.append(float(x[-1]))

        if len(final_xs) == 0:
            print(f"[WARN] No curves to plot for dataset '{dataset}'. Skipping.")
            continue

        # Fairness cutoff: smallest final x across optimizers for this dataset
        x_cutoff = min(final_xs)

        # -------------------------
        # PASS 2: plot all curves truncated to x_cutoff
        # -------------------------
        plt.figure(figsize=(6,4))  # default aspect ratio like your other script
        plt.title(PLOT_TITLE)

        for opt_name in ordered_opt_names:
            if opt_name not in curves:
                continue
            x, y, y_std = curves[opt_name]

            # Truncate THIS curve to the dataset cutoff
            keep = x <= x_cutoff
            if not np.any(keep):
                continue

            x_p = x[keep]
            y_p = y[keep]
            ystd_p = y_std[keep]

            label = OPT_LABEL.get(opt_name.lower(), opt_name)
            plt.plot(x_p, y_p, label=label)
            plt.fill_between(x_p, y_p - ystd_p, y_p + ystd_p, alpha=0.2)

        plt.xlim(0.0, x_cutoff)
        plt.xlabel("Cumulative FLOPs")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()

        out_path = os.path.join(OUTPUT_DIR, f"{plot_prefix}_{dataset}.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Saved {out_path} (x_cutoff={x_cutoff:.3e})")


if __name__ == "__main__":
    make_true_flops_plots(CSV_PATHS)

