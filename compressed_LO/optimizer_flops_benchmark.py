#!/usr/bin/env python3
"""
Optimizer FLOPs benchmark:

- Compares learned optimizers (CeLO, pruned CeLO, quantized CeLO)
  against conventional optimizers (Adam, Adafactor, SGD).
- Uses TransformerLM_LM1B_MultiRuntime_0 as the task.
- X-axis: cumulative FLOPs proxy
- Y-axis: training loss (EWMA smoothed)

Config is at the top (no CLI).
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Iterable

import csv

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np
import optax
import pandas as pd
import matplotlib.pyplot as plt

from learned_optimization.tasks.fixed.transformer_lm import (
    TransformerLM_LM1B_MultiRuntime_0,
)

from celo.factory import get_optimizer
from celo.utils import load_state

# Your helpers from compress_celo.py
from compress_celo import (
    build_celo_from_ckpt,
)

# --------------------------- Config ---------------------------------


@dataclass
class Cfg:
    # ---- Paths ----
    celo_ckpt: str = "./models/theta.state"  # main CeLO checkpoint (phase-2)

    # ---- Training ----
    num_steps: int = 2_000
    eval_every: int = 50
    seed: int = 7
    num_seeds: int = 1

    # ---- CeLO compression ----
    prune_sparsity: float = 0.5      # e.g., 50% sparsity
    include_pruned: bool = True
    include_quant8: bool = True

    # ---- FLOPs proxy parameters ----
    # Bits for parameters
    model_bits: int = 32        # model weights assumed float32
    celo_bits: int = 32         # CeLO theta weights float32
    celo_quant_bits: int = 8    # quantized CeLO theta
    # Optimizer overhead factors (very rough proxy)
    # We model: FLOPs_step = model_flops + optimizer_overhead_flops
    adam_overhead_factor: float = 2.0      # relative to model params
    adafactor_overhead_factor: float = 1.5
    sgd_overhead_factor: float = 1.0
    celo_overhead_factor: float = 1.0      # on CeLO theta params

    # ---- Smoothing ----
    ewm_span: int = 50  # EWMA span for smoothing plots

    # ---- Output ----
    csv_path: str = "optimizer_flops_benchmark.csv"
    plot_prefix: str = "optimizer_flops_benchmark"


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


def count_params(tree) -> int:
    """Count total number of scalar parameters in a pytree."""
    total = 0
    for x in flatten_params(tree):
        if hasattr(x, "size"):
            total += int(x.size)
    return total


def clone_tree(x):
    return jtu.tree_map(lambda a: a, x)


# ---------------------- Optax adapter -------------------------------


class OptaxAdapter:
    """Wrap an optax optimizer in the CeLO-like interface."""

    def __init__(self, tx: optax.GradientTransformation):
        self.tx = tx

    def init(self, params, model_state=None, num_steps=0):
        del model_state, num_steps
        return (params, self.tx.init(params))

    def get_params(self, state):
        params, _ = state
        return params

    def update(self, state, grad, loss=None):
        del loss
        params, opt_state = state
        updates, new_opt_state = self.tx.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state)


# ----------------- Build optimizers & FLOPs proxies -----------------


def build_optimizers_and_flops(cfg: Cfg, init_params) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Build CeLO variants + conventional optimizers and assign a FLOPs proxy
    per training step.

    FLOPs proxy (very rough, but consistent across optimizers):
        model_flops = N_model * model_bits
        optimizer_overhead:
            - CeLO:     N_theta * celo_bits * celo_overhead_factor
            - CeLO pruned: same but scaled by (1 - sparsity)
            - CeLO quant8: N_theta * celo_quant_bits * celo_overhead_factor
            - Adam:    N_model * model_bits * adam_overhead_factor
            - Adafactor: N_model * model_bits * adafactor_overhead_factor
            - SGD:     N_model * model_bits * sgd_overhead_factor

        total_flops_per_step = model_flops + optimizer_overhead
    """
    optimizers: Dict[str, Any] = {}
    flops_per_step: Dict[str, float] = {}

    # ----- Model FLOPs -----
    n_model = count_params(init_params)
    model_flops = float(max(n_model * cfg.model_bits, 1.0))
    print(f"[INFO] Model params: {n_model:,}")
    print(f"[INFO] Model FLOPs proxy per step: {model_flops:,.0f} (N_model * {cfg.model_bits} bits)")

    # ----- CeLO theta FLOPs -----
    lopt_theta = get_optimizer("celo")
    theta_template = lopt_theta.init(jax.random.PRNGKey(0))
    theta_base = load_state(cfg.celo_ckpt, theta_template)
    n_theta = count_params(theta_base)
    if n_theta == 0:
        print("[WARN] CeLO theta has 0 parameters; CeLO FLOPs proxy will be degenerate.")
    celo_theta_flops = float(max(n_theta * cfg.celo_bits * cfg.celo_overhead_factor, 1.0))

    print(f"[INFO] CeLO theta params: {n_theta:,}")
    print(
        f"[INFO] CeLO optimizer FLOPs proxy per step (baseline): "
        f"{celo_theta_flops:,.0f} (N_theta * {cfg.celo_bits} bits * factor={cfg.celo_overhead_factor})"
    )

    # ----- CeLO baseline -----
    _, opt_celo = build_celo_from_ckpt(cfg.celo_ckpt, variant="baseline")
    optimizers["celo"] = opt_celo
    flops_per_step["celo"] = model_flops + celo_theta_flops

    # ----- CeLO pruned -----
    if cfg.include_pruned:
        keep_ratio = 1.0 - cfg.prune_sparsity
        _, opt_pruned = build_celo_from_ckpt(
            cfg.celo_ckpt, variant="pruned", sparsity=cfg.prune_sparsity
        )
        optimizers["celo_pruned"] = opt_pruned
        pruned_theta_flops = celo_theta_flops * keep_ratio
        flops_per_step["celo_pruned"] = model_flops + pruned_theta_flops
        print(
            f"[INFO] CeLO pruned: sparsity={cfg.prune_sparsity:.2f}, "
            f"keep_ratio={keep_ratio:.2f}, "
            f"theta FLOPs≈{pruned_theta_flops:,.0f}"
        )

    # ----- CeLO quantized -----
    if cfg.include_quant8:
        _, opt_q8 = build_celo_from_ckpt(cfg.celo_ckpt, variant="quant8")
        optimizers["celo_q8"] = opt_q8
        q8_theta_flops = float(max(n_theta * cfg.celo_quant_bits * cfg.celo_overhead_factor, 1.0))
        flops_per_step["celo_q8"] = model_flops + q8_theta_flops
        print(
            f"[INFO] CeLO quant8: bits={cfg.celo_quant_bits}, "
            f"theta FLOPs≈{q8_theta_flops:,.0f}"
        )

    # ----- Conventional optimizers (Optax) -----
    # Adam
    opt_adam = OptaxAdapter(optax.adam(learning_rate=3e-4, b1=0.9, b2=0.999))
    optimizers["adam"] = opt_adam
    adam_overhead = n_model * cfg.model_bits * cfg.adam_overhead_factor
    flops_per_step["adam"] = model_flops + adam_overhead
    print(
        f"[INFO] Adam overhead FLOPs per step≈{adam_overhead:,.0f} "
        f"(factor={cfg.adam_overhead_factor})"
    )

    # Adafactor
    opt_adafactor = OptaxAdapter(optax.adafactor(learning_rate=3e-4))
    optimizers["adafactor"] = opt_adafactor
    adaf_overhead = n_model * cfg.model_bits * cfg.adafactor_overhead_factor
    flops_per_step["adafactor"] = model_flops + adaf_overhead
    print(
        f"[INFO] Adafactor overhead FLOPs per step≈{adaf_overhead:,.0f} "
        f"(factor={cfg.adafactor_overhead_factor})"
    )

    # SGD + momentum
    opt_sgd = OptaxAdapter(optax.sgd(learning_rate=1e-2, momentum=0.9))
    optimizers["sgd"] = opt_sgd
    sgd_overhead = n_model * cfg.model_bits * cfg.sgd_overhead_factor
    flops_per_step["sgd"] = model_flops + sgd_overhead
    print(
        f"[INFO] SGD overhead FLOPs per step≈{sgd_overhead:,.0f} "
        f"(factor={cfg.sgd_overhead_factor})"
    )

    return optimizers, flops_per_step


# ----------------- Single-seed run (all optimizers) -----------------


def run_single_seed(cfg: Cfg, seed: int, csv_writer) -> pd.DataFrame:
    """
    Run the FLOPs benchmark for a single random seed.
    Returns a DataFrame with columns:
        seed, optimizer, step, train_loss, cum_flops
    """

    # Task + data for THIS seed
    base_key, task, init_params, init_state = make_task_and_init(seed)
    train_stream = task.datasets.train

    # We want both loss+grad and a clean eval loss
    loss_and_grad_fn, _ = make_step_fns(task)

    # Build all optimizers + FLOP proxies
    optimizers, flops_per_step = build_optimizers_and_flops(cfg, init_params)
    names = list(optimizers.keys())

    # Initialize optimizer states (independent copy per optimizer)
    params0 = clone_tree(init_params)
    state0 = clone_tree(init_state)
    states: Dict[str, Any] = {}
    for name, opt in optimizers.items():
        states[name] = opt.init(
            clone_tree(params0), model_state=clone_tree(state0), num_steps=cfg.num_steps
        )

    # History for this seed
    history: Dict[str, Dict[str, list]] = {
        n: {"steps": [], "train_loss": [], "cum_flops": []} for n in names
    }
    last_train_loss: Dict[str, float] = {}
    cum_flops_tracker: Dict[str, float] = {n: 0.0 for n in names}

    # ---- Initial loss at step 0 ----
    init_batch = next(train_stream)
    for idx, name in enumerate(names):
        opt = optimizers[name]
        st = states[name]
        params = (
            st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
        )

        k0 = jax.random.fold_in(base_key, 0xBEEF + idx)
        tr_loss, _ = loss_and_grad_fn(params, k0, init_batch)
        tr_loss = float(tr_loss)

        last_train_loss[name] = tr_loss
        history[name]["steps"].append(0)
        history[name]["train_loss"].append(tr_loss)
        history[name]["cum_flops"].append(0.0)

        if csv_writer is not None:
            csv_writer.writerow([seed, name, 0, tr_loss, 0.0])

    # ---- Training loop ----
    main_key = base_key
    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        # training step for each optimizer
        for idx, name in enumerate(names):
            opt = optimizers[name]
            st = states[name]

            k = jax.random.fold_in(main_key, step * 997 + idx)
            params = (
                st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
            )
            tr_loss, grad = loss_and_grad_fn(params, k, batch)

            st = opt.update(st, grad, loss=tr_loss)
            states[name] = st
            tr_loss_val = float(tr_loss)
            last_train_loss[name] = tr_loss_val

            # accumulate FLOPs proxy
            cum_flops_tracker[name] += flops_per_step[name]

        # logging
        if step % cfg.eval_every == 0:
            for name in names:
                tl = last_train_loss[name]
                cf = cum_flops_tracker[name]

                history[name]["steps"].append(step)
                history[name]["train_loss"].append(tl)
                history[name]["cum_flops"].append(cf)

                if csv_writer is not None:
                    csv_writer.writerow([seed, name, step, tl, cf])

            msg = " | ".join(
                [f"{name}: tr={last_train_loss[name]:.4f}" for name in names]
            )
            print(f"[seed {seed} step {step:05d}] {msg}")

    # ---- Convert history to a DataFrame ----
    records = []
    for name in names:
        h = history[name]
        for stp, tl, cf in zip(h["steps"], h["train_loss"], h["cum_flops"]):
            records.append(
                dict(
                    seed=seed,
                    optimizer=name,
                    step=stp,
                    train_loss=tl,
                    cum_flops=cf,
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
            ["seed", "optimizer", "step", "train_loss", "cum_flops"]
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
    # Group by (optimizer, step) across seeds
    grouped = (
        df.groupby(["optimizer", "step"])
        .agg(
            mean_loss=("train_loss", "mean"),
            std_loss=("train_loss", "std"),
            mean_cum_flops=("cum_flops", "mean"),
        )
        .reset_index()
    )

    grouped["std_loss"] = grouped["std_loss"].fillna(0.0)
    grouped["stderr_loss"] = grouped["std_loss"] / np.sqrt(cfg.num_seeds)
    span = cfg.ewm_span

    # ------------- Plot: mean ± stderr vs FLOPs ----------------------
    plt.figure(figsize=(8, 5))
    for name in sorted(grouped["optimizer"].unique()):
        g = grouped[grouped["optimizer"] == name].sort_values("mean_cum_flops")

        g["mean_loss_ewm"] = g["mean_loss"].ewm(span=span, adjust=False).mean()
        g["stderr_ewm"] = g["stderr_loss"].ewm(span=span, adjust=False).mean()

        x = g["mean_cum_flops"].to_numpy()
        y = g["mean_loss_ewm"].to_numpy()
        err = g["stderr_ewm"].to_numpy()

        plt.plot(x, y, label=f"{name}")
        plt.fill_between(x, y - err, y + err, alpha=0.2)

    plt.xlabel("Cumulative FLOPs proxy")
    plt.ylabel("Training loss (EWMA of mean)")
    plt.title(
        f"Optimizers: loss vs FLOPs proxy (EWMA span={span}, {cfg.num_seeds} seed(s))"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    flops_plot = f"{cfg.plot_prefix}_loss_vs_flops_ewm.png"
    plt.savefig(flops_plot, dpi=300)
    print(f"[INFO] Saved {flops_plot}")


if __name__ == "__main__":
    main()

