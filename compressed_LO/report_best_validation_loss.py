import pandas as pd

DATASET_TASKS = [
    "TransformerLM_LM1B_MultiRuntime_0",
    "RNNLM_lm1bbytes_Patch32_LSTM128_Embed64",
    "RNNLM_lm1b32k_Patch32_LSTM256_Embed128",
    "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128",
]

OPTIMIZERS = ["celo", "celo_sam", "adam", "sgd", "adafactor"]

file = "./multi_dataset_optimizer_benchmark.csv"


def pick_col(cols, candidates, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(cols)}"
        )
    return None


df = pd.read_csv(file)
cols = df.columns

loss_col = pick_col(cols, ["train_loss"], required=True)

step_col = pick_col(
    cols,
    ["step", "global_step", "iter", "iteration", "train_step"],
    required=False,
)

# Optional: if you have multiple runs/seeds
seed_col = pick_col(cols, ["seed", "run", "trial"], required=False)

work = df.copy()
work = work[work["dataset"].isin(DATASET_TASKS)]
work = work[work["optimizer"].isin(OPTIMIZERS)]
work = work.dropna(subset=[loss_col])

group_keys = ["dataset", "optimizer"]
if seed_col is not None:
    group_keys.append(seed_col)

# index of minimum train_loss per group
idx = work.groupby(group_keys)[loss_col].idxmin()
best = work.loc[idx].copy()

keep_cols = group_keys + [loss_col]
if step_col is not None:
    keep_cols.append(step_col)

best = best[keep_cols].sort_values(group_keys).reset_index(drop=True)

print(f"Using loss column: {loss_col}")
if step_col is not None:
    print(f"Using step column: {step_col}")
if seed_col is not None:
    print(f"Using seed/run column: {seed_col}")

for task in DATASET_TASKS:
    print("\n" + "=" * 80)
    print(task)
    print(best[best["dataset"] == task].to_string(index=False))

out_path = "./best_train_loss_summary.csv"
best.to_csv(out_path, index=False)
print(f"\nWrote summary to: {out_path}")

