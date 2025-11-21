#!/usr/bin/env python3
"""
CeLO compression sweep:
- Baseline CeLO vs various pruned variants (different sparsity levels)
- Optional quantized CeLO (fake int8) as another compression point
Produces:
  1) training loss vs training iterations (EWMA-smoothed)
  2) training loss vs approximate FLOPs (EWMA-smoothed)

All parameters are configured in the Cfg dataclass below.
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
from celo.factory import get_optimizer
from celo.utils import load_state

# Your helpers & variants live in this module:
from compress_celo import (
    build_celo_from_ckpt,
    prune_by_magnitude,  # still imported if you want to poke at it
    quantize_fake_int8,  # same
)


# --------------------------- Config ---------------------------------


@dataclass
class Cfg:
    # ---- Paths ----
    celo_ckpt: str = "./theta.state"  # main CeLO checkpoint (phase-2)

    # ---- Training ----
    num_steps: int = 2_000
    eval_every: int = 1
    seed: int = 7
    num_seeds: int = 1

    # ---- Compression sweep ----
    # Sparsity levels for pruning (0.0 == baseline, so we skip 0.0 in the sweep)
    prune_levels: Tuple[float, ...] = (0.0, 0.5, 0.75, 0.999)
    include_quant8: bool = True  # include a pure quantized CeLO point

    # Bit-width assumptions for FLOP-ish accounting
    base_bits: int = 32  # baseline CeLO weights
    quant_bits: int = 8  # quant8 weights

    # ---- Smoothing ----
    ewm_span: int = 50  # EWMA span for smoothing plots

    # ---- Output ----
    csv_path: str = "celo_compression_sweep.csv"
    plot_prefix: str = "celo_compression"


CFG = Cfg()


# --------------------- Helper functions -----------------------------


def make_task_and_init(seed: int):
    """Create the LM task and initialize model parameters/state."""
    key = jax.random.PRNGKey(seed)
    task = TransformerLM_LM1B_MultiRuntime_0()
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
    *conceptual* FLOPs proxy per training step based on compression level.

    FLOPs proxy:
        baseline: N * base_bits
        prune s: N * (1 - s) * base_bits
        quant8:  N * quant_bits
    where N is total number of optimizer parameters.
    """
    # Load baseline theta once (matches build_celo_from_ckpt internals).
    lopt = get_optimizer("celo")
    theta_template = lopt.init(jax.random.PRNGKey(0))
    theta_base = load_state(cfg.celo_ckpt, theta_template)

    # Robustly count total params: anything array-like with a .size attribute
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

    lopt = get_optimizer("celo")
    theta_tmpl = lopt.init(jax.random.PRNGKey(0))
    theta_full = load_state(cfg.celo_ckpt, theta_tmpl)
    theta_pruned = prune_by_magnitude(theta_full, sparsity=1.0)

    print("baseline sparsity:", frac_zeros(theta_full))
    print("pruned sparsity  :", frac_zeros(theta_pruned))



    # ---- Pruned variants ----
    for sparsity in cfg.prune_levels:
        if sparsity <= 0.0:
            continue  # 0.0 == baseline

        keep_ratio = 1.0 - sparsity
        name = f"prune_{int(sparsity * 100)}"

        # conceptual nnz and FLOPs proxy
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

    # ---- Quantized variant (fake int8) ----
    if cfg.include_quant8:
        name_q = "quant8"
        nnz_q = total_params  # no pruning, just smaller bits
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

def run_single_seed(cfg: Cfg, seed: int, csv_writer) -> pd.DataFrame:
    """
    Run the compression sweep for a single random seed.
    Returns a DataFrame with columns:
        seed, variant, step, train_loss, cum_flops, compression
    """

    # Task + data for THIS seed
    base_key, task, init_params, init_state = make_task_and_init(seed)
    train_stream = task.datasets.train

    # We want both loss+grad and a clean eval loss
    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    # Build CeLO variants + FLOP proxies
    optimizers, flops_per_step, compression_ratio = build_celo_variants_and_flops(cfg)
    names = list(optimizers.keys())

    # Initialize optimizer states (independent copy per variant)
    states: Dict[str, Any] = {}
    for name, opt in optimizers.items():
        states[name] = opt.init(
            init_params,
            model_state=init_state,
            num_steps=cfg.num_steps,
        )


    # # Initialize optimizer states (independent copy per variant)
    # states: Dict[str, Any] = {}
    # for name, opt in optimizers.items():
    #     # This call should match whatever you already do in main()
    #     states[name] = opt.init(
    #         init_params,
    #         model_state=init_state,
    #         num_steps=cfg.num_steps,
    #         key=base_key,
    #         data=train_stream,
    #     )

    # History for this seed
    history: Dict[str, Dict[str, list]] = {
        n: {"steps": [], "train_loss": [], "cum_flops": []} for n in names
    }
    last_train_loss: Dict[str, float] = {}

    # ---- Initial evaluation at step 0 (no updates yet) ----
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
                [seed, name, 0, float(init_loss), 0.0, compression_ratio[name]]
            )

    # ---- Training loop for this seed ----
    main_key = base_key
    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        # training step for each variant
        for idx, name in enumerate(names):
            opt = optimizers[name]
            st = states[name]

            k = jax.random.fold_in(main_key, step * 997 + idx)
            params = opt.get_params(st)
            tr_loss, grad = loss_and_grad_fn(params, k, batch)

            st = opt.update(st, grad, loss=tr_loss)
            states[name] = st
            last_train_loss[name] = float(tr_loss)

        # logging for this seed
        if step % cfg.eval_every == 0:
            for name in names:
                tl = last_train_loss[name]
                cf = step * flops_per_step[name]

                history[name]["steps"].append(step)
                history[name]["train_loss"].append(tl)
                history[name]["cum_flops"].append(cf)

                if csv_writer is not None:
                    csv_writer.writerow(
                        [seed, name, step, tl, cf, compression_ratio[name]]
                    )

    # ---- Convert this seed's history to a DataFrame ----
    records = []
    for name in names:
        comp = compression_ratio[name]
        h = history[name]
        for stp, tl, cf in zip(h["steps"], h["train_loss"], h["cum_flops"]):
            records.append(
                dict(
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

    # CSV will contain all seeds together
    with open(cfg.csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["seed", "variant", "step", "train_loss", "cum_flops", "compression"]
        )

        for s in seeds:
            print(f"[INFO] Running seed {s}")
            df_seed = run_single_seed(cfg, s, csv_writer)
            all_dfs.append(df_seed)

    if not all_dfs:
        print("[WARN] No runs completed; nothing to plot.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    if df.empty:
        print("[WARN] No history recorded; no plots will be generated.")
        return

    # ----------------------- Aggregate stats -------------------------
    # Group by (variant, step) across seeds
    grouped = (
        df.groupby(["variant", "step"])
        .agg(
            mean_loss=("train_loss", "mean"),
            std_loss=("train_loss", "std"),
            mean_cum_flops=("cum_flops", "mean"),
            compression=("compression", "mean"),
        )
        .reset_index()
    )

    # Std might be NaN if there's only one seed; fill with 0 to be safe
    grouped["std_loss"] = grouped["std_loss"].fillna(0.0)

    # stderr = std / sqrt(num_seeds)
    grouped["stderr_loss"] = grouped["std_loss"] / np.sqrt(cfg.num_seeds)

    span = cfg.ewm_span

    # ----------------- Plot 1: mean ± stderr vs step -----------------
    plt.figure(figsize=(8, 5))
    for name in grouped["variant"].unique():
        g = grouped[grouped["variant"] == name].sort_values("step")

        # EWMA smoothing on the mean and error
        g["mean_loss_ewm"] = g["mean_loss"].ewm(span=span, adjust=False).mean()
        g["stderr_ewm"] = g["stderr_loss"].ewm(span=span, adjust=False).mean()

        x = g["step"].to_numpy()
        y = g["mean_loss_ewm"].to_numpy()
        err = g["stderr_ewm"].to_numpy()

        plt.plot(x, y, label=f"{name} (mean)")
        plt.fill_between(x, y - err, y + err, alpha=0.2)

    plt.xlabel("Training step")
    plt.ylabel("Training loss (EWMA of mean)")
    plt.title(f"CeLO compression sweep: mean ± stderr over {cfg.num_seeds} seeds (span={span})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    step_plot = f"{cfg.plot_prefix}_mean_pm_stderr_vs_step_ewm.png"
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
    plt.title(f"CeLO compression sweep: mean ± std vs FLOPs over {cfg.num_seeds} seeds (span={span})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    flops_plot = f"{cfg.plot_prefix}_mean_pm_std_vs_flops_ewm.png"
    plt.savefig(flops_plot, dpi=300)
    print(f"[INFO] Saved {flops_plot}")


if __name__ == "__main__":
    main()
