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

from learned_optimization.tasks.fixed.transformer_lm import (
    TransformerLM_LM1B_MultiRuntime_0,
)
from celo.factory import get_optimizer
from celo.utils import load_state

# Your helpers & variants live in this module:
from compress_celo import (
    build_celo_from_ckpt,
    prune_by_magnitude,     # still imported if you want to poke at it
    quantize_fake_int8,     # same
)


# --------------------------- Config ---------------------------------


@dataclass
class Cfg:
    # ---- Paths ----
    celo_ckpt: str = "./theta.state"  # main CeLO checkpoint (phase-2)

    # ---- Training ----
    num_steps: int = 2_000
    eval_every: int = 100
    seed: int = 7

    # ---- Compression sweep ----
    # Sparsity levels for pruning (0.0 == baseline, so we skip 0.0 in the sweep)
    prune_levels: Tuple[float, ...] = (0.0, 0.5, 0.75, 0.9)
    include_quant8: bool = True  # include a pure quantized CeLO point

    # Bit-width assumptions for FLOP-ish accounting
    base_bits: int = 32   # baseline CeLO weights
    quant_bits: int = 8   # quant8 weights

    # ---- Smoothing ----
    ewm_span: int = 50    # EWMA span for smoothing plots

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


# ----------------------------- Main ---------------------------------


def main():
    cfg = CFG

    # Task + data
    base_key, task, init_params, init_state = make_task_and_init(cfg.seed)
    train_stream = task.datasets.train
    val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train

    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    # Build CeLO variants + FLOP proxies
    optimizers, flops_per_step, compression_ratio = build_celo_variants_and_flops(cfg)
    names = list(optimizers.keys())

    # Initialize optimizer states (independent copies of params/state)
    states: Dict[str, Any] = {}
    for name, opt in optimizers.items():
        states[name] = opt.init(init_params, model_state=init_state, num_steps=cfg.num_steps)

    # History: one list per variant
    history: Dict[str, Dict[str, list]] = {
        n: {"steps": [], "train_loss": [], "cum_flops": []} for n in names
    }

    # Keep last training loss per variant so we can log it at eval points
    last_train_loss: Dict[str, float] = {n: float("nan") for n in names}

    # CSV writer (one row per (step, variant) at eval points)
    csv_file = open(cfg.csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["variant", "step", "train_loss", "val_loss", "cum_flops"])

    # Simple JIT warmup (to avoid counting compilation in timing, if you care)
    warm_batch = next(train_stream)
    for idx, n in enumerate(names):
        k_warm = jax.random.fold_in(base_key, 0xC0FFEE + idx)
        opt = optimizers[n]
        st = states[n]
        params = opt.get_params(st)
        _ = loss_and_grad_fn(params, k_warm, warm_batch)

    # Training loop
    start_wall = time.perf_counter()
    main_key = base_key

    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        # --- Training step for each variant ---
        for idx, name in enumerate(names):
            opt = optimizers[name]
            st = states[name]

            # Per-variant RNG
            k = jax.random.fold_in(main_key, step * 997 + idx)

            params = opt.get_params(st)
            tr_loss, grad = loss_and_grad_fn(params, k, batch)
            st = opt.update(st, grad, loss=tr_loss)
            states[name] = st
            last_train_loss[name] = float(tr_loss)

        # --- Evaluation / logging ---
        if step % cfg.eval_every == 0:
            try:
                val_batch = next(val_stream)
            except StopIteration:
                val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train
                val_batch = next(val_stream)

            log_strings = []
            for idx, name in enumerate(names):
                opt = optimizers[name]
                st = states[name]
                val_key = jax.random.fold_in(main_key, 100_000 + step * 13 + idx)

                val_params = opt.get_params(st)
                val_loss = eval_loss_fn(val_params, val_key, val_batch)
                train_loss = last_train_loss[name]
                cum_flops = step * flops_per_step[name]

                history[name]["steps"].append(step)
                history[name]["train_loss"].append(train_loss)
                history[name]["cum_flops"].append(cum_flops)

                csv_writer.writerow([name, step, train_loss, val_loss, cum_flops])
                log_strings.append(
                    f"{name}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            elapsed = time.perf_counter() - start_wall
            log_str = " | ".join(log_strings)
            print(f"[step {step:05d}] {log_str}   (elapsed {elapsed:.1f}s)")

    csv_file.close()
    print(f"[INFO] CSV written to {cfg.csv_path}")

    # ----------------------- Build DataFrame -------------------------

    records = []
    for name in names:
        steps = history[name]["steps"]
        train_losses = history[name]["train_loss"]
        cum_flops = history[name]["cum_flops"]
        comp = compression_ratio[name]
        for s, tl, cf in zip(steps, train_losses, cum_flops):
            records.append(
                dict(
                    variant=name,
                    step=s,
                    train_loss=tl,
                    cum_flops=cf,
                    compression=comp,
                )
            )

    df = pd.DataFrame(records)
    if df.empty:
        print("[WARN] No history recorded; no plots will be generated.")
        return

    # ----------------------- Plots (EWMA) ----------------------------

    span = cfg.ewm_span

    # 1) training loss vs iterations (EWMA)
    plt.figure()
    for name in names:
        sub = df[df["variant"] == name].sort_values("step")
        if sub.empty:
            continue
        smoothed = sub["train_loss"].ewm(span=span, adjust=False).mean()
        plt.plot(
            sub["step"],
            smoothed,
            label=f"{name} (x{compression_ratio[name]:.2f})",
        )

    plt.xlabel("Training step")
    plt.ylabel("Training loss (EWMA)")
    plt.title(f"CeLO compression sweep: training loss vs iterations (span={span})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    iters_plot = f"{cfg.plot_prefix}_train_loss_vs_iters_ewm.png"
    plt.savefig(iters_plot, dpi=300)
    print(f"[INFO] Saved {iters_plot}")

    # 2) training loss vs FLOPs (EWMA)
    plt.figure()
    for name in names:
        sub = df[df["variant"] == name].sort_values("cum_flops")
        if sub.empty:
            continue
        smoothed = sub["train_loss"].ewm(span=span, adjust=False).mean()
        plt.plot(
            sub["cum_flops"],
            smoothed,
            label=f"{name} (x{compression_ratio[name]:.2f})",
        )

    plt.xlabel("Cumulative FLOPs proxy (N * bits * steps)")
    plt.ylabel("Training loss (EWMA)")
    plt.title(f"CeLO compression sweep: training loss vs FLOPs (span={span})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    flops_plot = f"{cfg.plot_prefix}_train_loss_vs_flops_ewm.png"
    plt.savefig(flops_plot, dpi=300)
    print(f"[INFO] Saved {flops_plot}")


if __name__ == "__main__":
    main()

