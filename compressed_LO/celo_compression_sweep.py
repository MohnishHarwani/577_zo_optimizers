#!/usr/bin/env python3
"""
CeLO compression sweep:
- Baseline CeLO vs various pruned variants (different sparsity levels)
- Optional quantized CeLO (fake int8) as another compression point
Runs on all four language modeling tasks:
    - TransformerLM_LM1B_MultiRuntime_0
    - RNNLM_lm1b32k_Patch32_LSTM256_Embed128
    - RNNLM_lm1bbytes_Patch32_LSTM128_Embed64
    - RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128

Produces:
  1) training loss vs training iterations (EWMA-smoothed)
  2) training loss vs approximate FLOPs (EWMA-smoothed)
For each dataset.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Iterable
import time
import csv

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from learned_optimization.tasks.fixed.transformer_lm import (
    TransformerLM_LM1B_MultiRuntime_0,
)
from learned_optimization.tasks.fixed.rnn_lm import (
    RNNLM_lm1b32k_Patch32_LSTM256_Embed128,
    RNNLM_lm1bbytes_Patch32_LSTM128_Embed64,
    RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128,
)

from celo.factory import get_optimizer
from celo.utils import load_state

from compress_celo import (
    build_celo_from_ckpt,
    prune_by_magnitude,
    quantize_fake_int8,
)

# --------------------------- Tasks ----------------------------------

DATASET_TASKS = {
    "TransformerLM_LM1B_MultiRuntime_0": TransformerLM_LM1B_MultiRuntime_0,
    "RNNLM_lm1b32k_Patch32_LSTM256_Embed128": RNNLM_lm1b32k_Patch32_LSTM256_Embed128,
    "RNNLM_lm1bbytes_Patch32_LSTM128_Embed64": RNNLM_lm1bbytes_Patch32_LSTM128_Embed64,
    "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128": RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128,
}


# --------------------------- Config ---------------------------------


@dataclass
class Cfg:
    # ---- Paths ----
    celo_ckpt: str = "./models/theta.state"  # main CeLO checkpoint (phase-2)

    # ---- Training ----
    num_steps: int = 2_000
    eval_every: int = 1
    seed: int = 7
    num_seeds: int = 1

    # ---- Compression sweep ----
    prune_levels: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95)
    include_quant8: bool = True
    include_randomized: bool = False

    # Bit-width assumptions for FLOP-ish accounting
    base_bits: int = 32
    quant_bits: int = 8

    # ---- Smoothing ----
    ewm_span: int = 35

    # ---- Output ----
    csv_path: str = "celo_compression_sweep_all_datasets.csv"
    plot_prefix: str = "celo_compression"


CFG = Cfg()


# --------------------- Helper functions -----------------------------


def make_task_and_init(dataset_name: str, seed: int):
    """Create the LM task and initialize model parameters/state for a given dataset."""
    if dataset_name not in DATASET_TASKS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    key = jax.random.PRNGKey(seed)
    task_ctor = DATASET_TASKS[dataset_name]
    task = task_ctor()
    key, k1 = jax.random.split(key)
    params, model_state = task.init_with_state(k1)
    return key, task, params, model_state


def to_jnp_tree(x):
    """Convert array-like leaves to jnp.array; leave scalars/None alone."""
    return jnp.asarray(x) if hasattr(x, "dtype") else x


def make_step_fns(task):
    """Create JITted step & eval functions that close over `task`."""

    def loss_fn(params, key, batch):
        batch = jax.tree.map(to_jnp_tree, batch)
        return task.loss(params, key, batch)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def eval_loss(params, key, batch):
        batch = jax.tree.map(to_jnp_tree, batch)
        return float(task.loss(params, key, batch))

    return loss_and_grad, eval_loss


def flatten_params(theta) -> Iterable:
    """Return all leaves in a tree as a flat iterable."""
    leaves, _ = jtu.tree_flatten(theta)
    return leaves


# ----------------- Build CeLO variants & FLOPs ----------------------


def build_celo_variants_and_flops(cfg: Cfg):
    """
    Build CeLO optimizer variants (baseline, pruned, quant8) and assign a
    conceptual FLOPs proxy per training step based on compression level.
    """
    lopt = get_optimizer("celo")
    theta_template = lopt.init(jax.random.PRNGKey(0))
    theta_base = load_state(cfg.celo_ckpt, theta_template)

    total_params = 0
    for x in flatten_params(theta_base):
        if hasattr(x, "size"):
            total_params += int(x.size)

    if total_params == 0:
        print("[WARN] total_params == 0; FLOPs proxy will be degenerate.")
    baseline_flops = float(max(total_params * cfg.base_bits, 1.0))

    print(f"[INFO] Baseline CeLO params: {total_params:,}")
    print(
        f"[INFO] Baseline FLOPs proxy per step: {baseline_flops:,.0f} "
        f"(N * {cfg.base_bits} bits)"
    )

    optimizers: Dict[str, Any] = {}
    flops_per_step: Dict[str, float] = {}
    compression_ratio: Dict[str, float] = {}

    # ---- Baseline (no compression) ----
    name_base = "celo"
    _, opt_base = build_celo_from_ckpt(cfg.celo_ckpt, variant="baseline")
    optimizers[name_base] = opt_base
    flops_per_step[name_base] = baseline_flops
    compression_ratio[name_base] = 1.0

    from jax import tree_util as jtu
    import jax.numpy as jnp

    def frac_zeros(tree) -> float:
        leaves, _ = jtu.tree_flatten(tree)
        total = 0
        zeros = 0
        for x in leaves:
            if hasattr(x, "size") and getattr(x, "ndim", 0) > 0 and x.size > 0:
                arr = jnp.asarray(x)
                total += int(arr.size)
                zeros += int((arr == 0).sum())
        if total == 0:
            print("[WARN] frac_zeros: no array leaves found; returning 0.0")
            return 0.0
        return float(zeros) / float(total)

    def randomize_pruned(theta, sparsity: float, key: jax.Array):
        """Same mask as prune_by_magnitude, but random values on pruned entries."""
        theta_pruned = prune_by_magnitude(theta, sparsity=sparsity)

        leaves_theta, treedef = jtu.tree_flatten(theta)
        leaves_pruned, _ = jtu.tree_flatten(theta_pruned)

        subkeys = jax.random.split(key, len(leaves_theta))
        new_leaves = []

        for w, p, k in zip(leaves_theta, leaves_pruned, subkeys):
            if not hasattr(w, "shape"):
                new_leaves.append(w)
                continue

            arr = jnp.asarray(w)
            parr = jnp.asarray(p)
            if arr.size == 0 or arr.ndim == 0:
                new_leaves.append(w)
                continue

            pruned_mask = (parr == 0)

            std = jnp.std(arr)
            std = jnp.where(std == 0.0, 1e-8, std)

            rand = jax.random.normal(k, arr.shape) * (std * 10)
            new_arr = arr * (~pruned_mask) + rand * pruned_mask
            new_leaves.append(new_arr)

        return jtu.tree_unflatten(treedef, new_leaves)

    # Simple sanity check print (can be removed)
    lopt_dbg = get_optimizer("celo")
    theta_tmpl_dbg = lopt_dbg.init(jax.random.PRNGKey(0))
    theta_full_dbg = load_state(cfg.celo_ckpt, theta_tmpl_dbg)
    theta_pruned_dbg = prune_by_magnitude(theta_full_dbg, sparsity=1.0)
    print("baseline sparsity:", frac_zeros(theta_full_dbg))
    print("pruned sparsity  :", frac_zeros(theta_pruned_dbg))

    # ---- Pruned variants ----
    for sparsity in cfg.prune_levels:
        if sparsity <= 0.0:
            continue

        keep_ratio = 1.0 - sparsity
        name = f"prune_{int(sparsity * 100)}"

        nnz_p = total_params * keep_ratio
        eff_flops = float(max(nnz_p * cfg.base_bits, 1.0))
        comp = baseline_flops / eff_flops if eff_flops > 0 else float("inf")

        print(
            f"[INFO] {name}: sparsity={sparsity:.2f} keep={keep_ratio:.2f}  "
            f"FLOPs≈{eff_flops:,.0f}  compression≈{comp:.2f}x"
        )

        _, opt_p = build_celo_from_ckpt(
            cfg.celo_ckpt,
            variant="pruned",
            sparsity=sparsity,
        )
        optimizers[name] = opt_p
        flops_per_step[name] = eff_flops
        compression_ratio[name] = comp

        # randomized variant disabled by default; hook preserved if you want it
        if cfg.include_randomized:
            name_rand = f"prune_{int(sparsity * 100)}_rand"
            rand_key = jax.random.PRNGKey(cfg.seed + int(sparsity * 1000))
            theta_rand = randomize_pruned(theta_base, sparsity=sparsity, key=rand_key)

            lopt_rand = get_optimizer("celo")
            opt_rand = lopt_rand.opt_fn(theta_rand)

            optimizers[name_rand] = opt_rand
            flops_per_step[name_rand] = eff_flops
            compression_ratio[name_rand] = comp

    # ---- Quantized variant (fake int8) ----
    if cfg.include_quant8:
        name_q = "quant8"
        nnz_q = total_params
        eff_flops_q = float(max(nnz_q * cfg.quant_bits, 1.0))
        comp_q = baseline_flops / eff_flops_q if eff_flops_q > 0 else float("inf")

        print(
            f"[INFO] {name_q}: bits={cfg.quant_bits}  "
            f"FLOPs≈{eff_flops_q:,.0f}  compression≈{comp_q:.2f}x"
        )

        _, opt_q = build_celo_from_ckpt(cfg.celo_ckpt, variant="quant8")
        optimizers[name_q] = opt_q
        flops_per_step[name_q] = eff_flops_q
        compression_ratio[name_q] = comp_q

    return optimizers, flops_per_step, compression_ratio


# ---------------------------- Runner --------------------------------


def run_single_seed(cfg: Cfg, seed: int, dataset_name: str, csv_writer) -> pd.DataFrame:
    """
    Run the compression sweep for a single random seed on a single dataset.
    Returns a DataFrame with columns:
        dataset, seed, variant, step, train_loss, cum_flops, compression
    """

    print(f"[INFO] Dataset={dataset_name}, seed={seed}")

    base_key, task, init_params, init_state = make_task_and_init(dataset_name, seed)
    train_stream = task.datasets.train

    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    optimizers, flops_per_step, compression_ratio = build_celo_variants_and_flops(cfg)
    names = list(optimizers.keys())

    states: Dict[str, Any] = {}
    for name, opt in optimizers.items():
        states[name] = opt.init(
            init_params,
            model_state=init_state,
            num_steps=cfg.num_steps,
        )

    history: Dict[str, Dict[str, list]] = {
        n: {"steps": [], "train_loss": [], "cum_flops": []} for n in names
    }
    last_train_loss: Dict[str, float] = {}

    # Initial eval at step 0
    init_batch = next(train_stream)
    for idx, name in enumerate(names):
        opt = optimizers[name]
        st = states[name]
        params = opt.get_params(st)

        k0 = jax.random.fold_in(base_key, 0xBEEF + idx)
        init_loss = eval_loss_fn(params, k0, init_batch)

        last_train_loss[name] = float(init_loss)
        history[name]["steps"].append(0)
        history[name]["train_loss"].append(float(init_loss))
        history[name]["cum_flops"].append(0.0)

        if csv_writer is not None:
            csv_writer.writerow(
                [dataset_name, seed, name, 0, float(init_loss), 0.0, compression_ratio[name]]
            )

    main_key = base_key
    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        for idx, name in enumerate(names):
            opt = optimizers[name]
            st = states[name]

            k = jax.random.fold_in(main_key, step * 997 + idx)
            params = opt.get_params(st)
            tr_loss, grad = loss_and_grad_fn(params, k, batch)

            st = opt.update(st, grad, loss=tr_loss)
            states[name] = st
            last_train_loss[name] = float(tr_loss)

        if step % cfg.eval_every == 0:
            for name in names:
                tl = last_train_loss[name]
                cf = step * flops_per_step[name]

                history[name]["steps"].append(step)
                history[name]["train_loss"].append(tl)
                history[name]["cum_flops"].append(cf)

                if csv_writer is not None:
                    csv_writer.writerow(
                        [dataset_name, seed, name, step, tl, cf, compression_ratio[name]]
                    )

    records = []
    for name in names:
        comp = compression_ratio[name]
        h = history[name]
        for stp, tl, cf in zip(h["steps"], h["train_loss"], h["cum_flops"]):
            records.append(
                dict(
                    dataset=dataset_name,
                    seed=seed,
                    variant=name,
                    step=stp,
                    train_loss=tl,
                    cum_flops=cf,
                    compression=comp,
                )
            )

    return pd.DataFrame(records)


# ----------------------------- Main ---------------------------------


def main():
    cfg = CFG

    seeds = [cfg.seed + i for i in range(cfg.num_seeds)]
    all_dfs = []

    with open(cfg.csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["dataset", "seed", "variant", "step", "train_loss", "cum_flops", "compression"]
        )

        for dataset_name in DATASET_TASKS.keys():
            for s in seeds:
                df_seed = run_single_seed(cfg, s, dataset_name, csv_writer)
                all_dfs.append(df_seed)

    if not all_dfs:
        print("[WARN] No runs completed; nothing to plot.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    if df.empty:
        print("[WARN] No history recorded; no plots will be generated.")
        return

    # ----------------------- Aggregate stats -------------------------
    span = cfg.ewm_span

    # One set of plots per dataset
    for dataset_name in sorted(df["dataset"].unique()):
        df_d = df[df["dataset"] == dataset_name].copy()

        grouped = (
            df_d.groupby(["variant", "step"])
            .agg(
                mean_loss=("train_loss", "mean"),
                std_loss=("train_loss", "std"),
                mean_cum_flops=("cum_flops", "mean"),
                compression=("compression", "mean"),
            )
            .reset_index()
        )
        grouped["std_loss"] = grouped["std_loss"].fillna(0.0)
        grouped["stderr_loss"] = grouped["std_loss"] / np.sqrt(cfg.num_seeds)

        # ----------------- Plot 1: mean ± stderr vs step -----------------
        plt.figure(figsize=(8, 5))
        for name in grouped["variant"].unique():
            g = grouped[grouped["variant"] == name].sort_values("step")

            g["mean_loss_ewm"] = g["mean_loss"].ewm(span=span, adjust=False).mean()
            g["stderr_ewm"] = g["stderr_loss"].ewm(span=span, adjust=False).mean()

            x = g["step"].to_numpy()
            y = g["mean_loss_ewm"].to_numpy()
            err = g["stderr_ewm"].to_numpy()

            plt.plot(x, y, label=f"{name} (mean)")
            plt.fill_between(x, y - err, y + err, alpha=0.2)

        plt.xlabel("Training step")
        plt.ylabel("Training loss (EWMA of mean)")
        plt.title(
            f"{dataset_name}: CeLO compression sweep "
            f"(mean ± stderr over {cfg.num_seeds} seeds, span={span})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        step_plot = f"{cfg.plot_prefix}_{dataset_name}_mean_pm_stderr_vs_step_ewm.pdf"
        plt.savefig(step_plot, dpi=300)
        print(f"[INFO] Saved {step_plot}")

        # ------------- Plot 2: mean ± std vs FLOPs ----------------------
        plt.figure(figsize=(8, 5))
        for name in grouped["variant"].unique():
            g = grouped[grouped["variant"] == name].sort_values("mean_cum_flops")

            g["mean_loss_ewm"] = g["mean_loss"].ewm(span=span, adjust=False).mean()
            g["std_ewm"] = g["std_loss"].ewm(span=span, adjust=False).mean()

            x = g["mean_cum_flops"].to_numpy()
            y = g["mean_loss_ewm"].to_numpy()
            err = g["std_ewm"].to_numpy()

            plt.plot(x, y, label=f"{name} (mean)")
            plt.fill_between(x, y - err, y + err, alpha=0.2)

        plt.xlabel("Cumulative FLOPs proxy (N * bits * steps)")
        plt.ylabel("Training loss (EWMA of mean)")
        plt.title(
            f"{dataset_name}: CeLO compression sweep "
            f"(mean ± std vs FLOPs over {cfg.num_seeds} seeds, span={span})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        flops_plot = f"{cfg.plot_prefix}_{dataset_name}_mean_pm_std_vs_flops_ewm.pdf"
        plt.savefig(flops_plot, dpi=300)
        print(f"[INFO] Saved {flops_plot}")


if __name__ == "__main__":
    main()

