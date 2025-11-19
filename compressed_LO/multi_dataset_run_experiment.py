#!/usr/bin/env python3
"""
Multi-dataset optimizer benchmark.

- Runs a single training run per NLP task/dataset.
- Uses the same optimizer set as run_experiment.py (CeLO, pruned CeLO, quant8, Adam, Adafactor, SGD,
  and optionally the two-stage CeLO2 set).
- Writes a long-format CSV with columns:
    dataset, step, optimizer, train_loss, val_loss
- Produces EWMA-smoothed training curves per dataset.
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Callable, OrderedDict

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax
import pandas as pd
import matplotlib.pyplot as plt

from learned_optimization.tasks.fixed.transformer_lm import (
    TransformerLM_LM1B_MultiRuntime_0,
)
from learned_optimization.tasks.fixed import rnn_lm

from compress_celo import build_celo_from_ckpt, build_celo_two_stage


# --------------------------- Config ---------------------------------


@dataclass
class RunCfg:
    # Training
    num_steps: int = 2_000
    eval_every: int = 100
    seed: int = 7

    # CeLO checkpoints
    celo_ckpt: str = "./theta.state"
    celo_phase1_ckpt: str = "./theta_phase1.state"
    celo_phase2_ckpt: str = "./theta_phase2.state"
    enable_two_stage_second_set: bool = True

    # Compression / pruning
    prune_sparsity: float = 0.50
    use_quant8: bool = True

    # Baseline optimizer hyperparameters
    adam_lr: float = 3e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adafactor_lr: float = 3e-4
    sgd_lr: float = 1e-2
    sgd_momentum: float = 0.9

    # Output
    csv_path: str = "multi_dataset_optimizer_benchmark.csv"
    plot_prefix: str = "multi_dataset_training"
    ewm_span: int = 50  # EWMA smoothing span for plots


CFG = RunCfg()


# ---------------------- Dataset / Task registry ---------------------


# Map "dataset name" -> task constructor
# You can comment out or add entries as you like.
DATASET_TASKS: "OrderedDict[str, Callable[[], Any]]" = {
    # Transformer LM on LM1B (same as your existing run_experiment)
    "TransformerLM_LM1B_MultiRuntime_0": TransformerLM_LM1B_MultiRuntime_0,
    # RNN LMs on LM1B / Wikipedia from learned_optimization.tasks.fixed.rnn_lm
    "RNNLM_lm1bbytes_Patch32_LSTM128_Embed64": (
        rnn_lm.RNNLM_lm1bbytes_Patch32_LSTM128_Embed64
    ),
    "RNNLM_lm1b32k_Patch32_LSTM256_Embed128": (
        rnn_lm.RNNLM_lm1b32k_Patch32_LSTM256_Embed128
    ),
    "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128": (
        rnn_lm.RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128
    ),
}


# ----------------------- Utility classes/fns ------------------------


class OptaxAdapter:
    """Wrap an optax optimizer in the CeLO-like interface used in run_experiment."""

    def __init__(self, tx: optax.GradientTransformation):
        self.tx = tx

    def init(self, params, model_state=None, num_steps=0):
        del model_state, num_steps
        return (params, self.tx.init(params))

    def get_params(self, state):
        params, _ = state
        return params

    def update(self, state, grad, loss=None):
        params, opt_state = state
        updates, new_opt_state = self.tx.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state)


def make_task_and_init(task_ctor: Callable[[], Any], seed: int):
    """Create task instance and initialize parameters/state."""
    key = jax.random.PRNGKey(seed)
    task = task_ctor()
    key, k1 = jax.random.split(key)
    params, model_state = task.init_with_state(k1)
    return key, task, params, model_state


def make_step_fns(task):
    """Return (loss_and_grad_fn, eval_loss_fn) that close over `task`."""

    def to_jnp_tree(x):
        # Convert anything array-like to jnp.array; leave scalars/None alone.
        return jnp.asarray(x) if hasattr(x, "dtype") else x

    def loss_fn(params, key, batch):
        batch = jax.tree.map(to_jnp_tree, batch)
        return task.loss(params, key, batch)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    def eval_loss(params, key, batch):
        batch = jax.tree.map(to_jnp_tree, batch)
        return float(task.loss(params, key, batch))

    return loss_and_grad, eval_loss


def clone_tree(x):
    return jtu.tree_map(lambda a: a, x)


# ------------------------ Single-dataset run -------------------------


def run_single_dataset(
    dataset_name: str,
    task_ctor: Callable[[], Any],
    cfg: RunCfg,
    csv_writer: csv.writer,
):
    """Run all optimizers on a single dataset/task and log to CSV."""

    print(f"\n=== Running dataset: {dataset_name} ===")

    base_key, task, init_params, init_state = make_task_and_init(task_ctor, cfg.seed)
    train_stream = task.datasets.train
    val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train

    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    # ---- Build optimizers (same as run_experiment) ----
    opt_objs: Dict[str, Any] = {}

    _, opt_celo = build_celo_from_ckpt(cfg.celo_ckpt, variant="baseline")
    _, opt_pruned = build_celo_from_ckpt(
        cfg.celo_ckpt, variant="pruned", sparsity=cfg.prune_sparsity
    )
    _, opt_quant8 = build_celo_from_ckpt(cfg.celo_ckpt, variant="quant8")

    opt_adam = OptaxAdapter(optax.adam(cfg.adam_lr, b1=cfg.adam_b1, b2=cfg.adam_b2))
    opt_adafactor = OptaxAdapter(optax.adafactor(learning_rate=cfg.adafactor_lr))
    opt_sgd = OptaxAdapter(
        optax.sgd(learning_rate=cfg.sgd_lr, momentum=cfg.sgd_momentum)
    )

    opt_objs["celo"] = opt_celo
    opt_objs["celo_prune"] = opt_pruned
    opt_objs["celo_q8"] = opt_quant8
    opt_objs["adam"] = opt_adam
    opt_objs["adafactor"] = opt_adafactor
    opt_objs["sgd"] = opt_sgd

    if cfg.enable_two_stage_second_set and cfg.celo_phase1_ckpt and cfg.celo_phase2_ckpt:
        _, opt_celo2 = build_celo_two_stage(
            cfg.celo_phase1_ckpt, cfg.celo_phase2_ckpt, variant="baseline"
        )
        _, opt_prune2 = build_celo_two_stage(
            cfg.celo_phase1_ckpt,
            cfg.celo_phase2_ckpt,
            variant="pruned",
            sparsity=cfg.prune_sparsity,
        )
        _, opt_quant2 = build_celo_two_stage(
            cfg.celo_phase1_ckpt, cfg.celo_phase2_ckpt, variant="quant8"
        )
        opt_objs["celo2"] = opt_celo2
        opt_objs["celo2_prune"] = opt_prune2
        opt_objs["celo2_q8"] = opt_quant2

    names = list(opt_objs.keys())

    # Independent copies of params/state per optimizer
    params0 = clone_tree(init_params)
    state0 = clone_tree(init_state)
    states: Dict[str, Any] = {}
    for name, opt in opt_objs.items():
        states[name] = opt.init(
            clone_tree(params0), model_state=clone_tree(state0), num_steps=cfg.num_steps
        )

    # Warmup to compile loss_and_grad
    warm_batch = next(train_stream)
    for idx, n in enumerate(names):
        k_warm = jax.random.fold_in(base_key, 0xC0FFEE + idx)
        opt, st = opt_objs[n], states[n]
        params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
        _ = loss_and_grad_fn(params, k_warm, warm_batch)

    main_key = base_key

    # ---- Training loop ----
    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        # Train each optimizer
        tr_losses: Dict[str, float] = {}
        for idx, n in enumerate(names):
            k = jax.random.fold_in(main_key, step * 997 + idx)
            opt = opt_objs[n]
            st = states[n]
            params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)

            tr_loss, grad = loss_and_grad_fn(params, k, batch)
            st = opt.update(st, grad, loss=tr_loss)
            states[n] = st
            tr_losses[n] = float(tr_loss)

        # Evaluation (val loss) every eval_every steps
        val_losses: Dict[str, float] = {n: float("nan") for n in names}
        if step % cfg.eval_every == 0:
            try:
                val_batch = next(val_stream)
            except StopIteration:
                val_stream = (
                    task.datasets.valid
                    if hasattr(task.datasets, "valid")
                    else task.datasets.train
                )
                val_batch = next(val_stream)

            for idx, n in enumerate(names):
                opt = opt_objs[n]
                st = states[n]
                val_params = (
                    st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
                )
                val_key = jax.random.fold_in(main_key, 100_000 + step * 13 + idx)
                v = eval_loss_fn(val_params, val_key, val_batch)
                val_losses[n] = float(v)

        # Log to CSV: one row per optimizer
        for n in names:
            csv_writer.writerow(
                [
                    dataset_name,
                    step,
                    n,  # optimizer name
                    tr_losses[n],
                    val_losses[n],
                ]
            )

        if step % cfg.eval_every == 0:
            msg = " | ".join(
                [f"{n}: tr={tr_losses[n]:.4f}, val={val_losses[n]:.4f}" for n in names]
            )
            print(f"[{dataset_name} step {step:05d}] {msg}")


# ----------------------- Plotting from CSV --------------------------


def make_plots(cfg: RunCfg):
    df = pd.read_csv(cfg.csv_path)
    span = cfg.ewm_span

    # Sort for consistent plotting
    df = df.sort_values(["dataset", "optimizer", "step"])

    # For each dataset, plot EWMA(train_loss) vs step, all optimizers
    datasets = df["dataset"].unique()
    for dataset in datasets:
        df_d = df[df["dataset"] == dataset]

        plt.figure(figsize=(8, 5))
        plt.title(f"{dataset}: Training Loss vs Step (EWMA span={span})")

        for opt_name in sorted(df_d["optimizer"].unique()):
            sub = df_d[df_d["optimizer"] == opt_name]
            sub = sub.sort_values("step")
            y_smooth = sub["train_loss"].ewm(span=span, adjust=False).mean()
            plt.plot(sub["step"], y_smooth, label=opt_name)

        plt.xlabel("Step")
        plt.ylabel("Training loss (cross-entropy)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f"{cfg.plot_prefix}_{dataset}_train_vs_step_ewm.png"
        plt.savefig(out_path, dpi=300)
        print(f"[PLOT] Saved {out_path}")


# ------------------------------ Main --------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_steps", type=int, default=CFG.num_steps)
    ap.add_argument("--eval_every", type=int, default=CFG.eval_every)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--csv_path", type=str, default=CFG.csv_path)
    ap.add_argument("--plot_prefix", type=str, default=CFG.plot_prefix)
    ap.add_argument("--celo_ckpt", type=str, default=CFG.celo_ckpt)
    ap.add_argument("--celo_phase1_ckpt", type=str, default=CFG.celo_phase1_ckpt)
    ap.add_argument("--celo_phase2_ckpt", type=str, default=CFG.celo_phase2_ckpt)
    ap.add_argument("--enable_two_stage_second_set", action="store_true", default=CFG.enable_two_stage_second_set)
    ap.add_argument("--prune_sparsity", type=float, default=CFG.prune_sparsity)
    ap.add_argument("--adam_lr", type=float, default=CFG.adam_lr)
    ap.add_argument("--adafactor_lr", type=float, default=CFG.adafactor_lr)
    ap.add_argument("--sgd_lr", type=float, default=CFG.sgd_lr)
    ap.add_argument("--sgd_momentum", type=float, default=CFG.sgd_momentum)
    ap.add_argument("--ewm_span", type=int, default=CFG.ewm_span)
    args = ap.parse_args()

    cfg = RunCfg(
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        csv_path=args.csv_path,
        plot_prefix=args.plot_prefix,
        celo_ckpt=args.celo_ckpt,
        celo_phase1_ckpt=args.celo_phase1_ckpt,
        celo_phase2_ckpt=args.celo_phase2_ckpt,
        enable_two_stage_second_set=args.enable_two_stage_second_set,
        prune_sparsity=args.prune_sparsity,
        adam_lr=args.adam_lr,
        adafactor_lr=args.adafactor_lr,
        sgd_lr=args.sgd_lr,
        sgd_momentum=args.sgd_momentum,
        ewm_span=args.ewm_span,
    )

    # Run all datasets and write one big CSV
    with open(cfg.csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["dataset", "step", "optimizer", "train_loss", "val_loss"])

        for dataset_name, task_ctor in DATASET_TASKS.items():
            run_single_dataset(dataset_name, task_ctor, cfg, writer)

    print(f"\n[INFO] All datasets complete. CSV written to {cfg.csv_path}")
    make_plots(cfg)


if __name__ == "__main__":
    main()

