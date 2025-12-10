#!/usr/bin/env python3
"""
Multi-dataset FLOPs benchmark.

- Runs all language modeling tasks in DATASET_TASKS
  (same set as multi_dataset_run_experiment.py).
- Compares:
    * CeLO (baseline)
    * Pruned CeLO
    * Quantized CeLO
    * Adam
    * Adafactor
    * SGD
- Logs training loss and a FLOPs *proxy*.
- Produces ONE plot per dataset:
    X-axis: cumulative FLOPs proxy
    Y-axis: EWMA-smoothed training loss
"""

from dataclasses import dataclass
from typing import Any, Dict, Callable, OrderedDict, Tuple, Iterable

import csv

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from learned_optimization.tasks.fixed.transformer_lm import (
    TransformerLM_LM1B_MultiRuntime_0,
)
from learned_optimization.tasks.fixed import rnn_lm

from compress_celo import build_celo_from_ckpt, build_celo_two_stage  # your helper :contentReference[oaicite:2]{index=2}


# --------------------------- Config ---------------------------------


@dataclass
class RunCfg:
    # Training
    num_steps: int = 2_000
    eval_every: int = 100
    seed: int = 7
    num_repeats: int = 5

    # CeLO checkpoints
    celo_ckpt: str = "./models/theta.state"
    celo_phase1_ckpt: str = "./models/theta_phase1.state"
    celo_phase2_ckpt: str = "./models/theta_phase2.state"
    enable_two_stage_second_set: bool = False  # set True if you want celo2 set

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

    # FLOPs proxy parameters
    model_bits: int = 32          # assume model params are float32
    celo_bits: int = 32           # CeLO theta bits (baseline)
    celo_quant_bits: int = 8      # quantized CeLO theta
    # Relative overhead factors for optimizers (very rough)
    adam_overhead_factor: float = 2.0
    adafactor_overhead_factor: float = 1.5
    sgd_overhead_factor: float = 1.0
    celo_overhead_factor: float = 1.0

    # Output
    csv_path: str = "multi_dataset_flops_benchmark.csv"
    plot_prefix: str = "multi_dataset_flops"
    ewm_span: int = 5


CFG = RunCfg()


# ---------------------- Dataset / Task registry ---------------------


# Same language modeling experiments as multi_dataset_run_experiment.py 
DATASET_TASKS: "OrderedDict[str, Callable[[], Any]]" = {
    # Transformer LM on LM1B
    "TransformerLM_LM1B_MultiRuntime_0": TransformerLM_LM1B_MultiRuntime_0,

    # RNN LMs on LM1B / Wikipedia
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


def make_task_and_init(task_ctor: Callable[[], Any], seed: int):
    key = jax.random.PRNGKey(seed)
    task = task_ctor()
    key, k1 = jax.random.split(key)
    params, model_state = task.init_with_state(k1)
    return key, task, params, model_state


def make_step_fns(task):
    """Return (loss_and_grad_fn, eval_loss_fn) that close over `task`."""

    def to_jnp_tree(x):
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


def flatten_params(tree) -> Iterable:
    leaves, _ = jtu.tree_flatten(tree)
    return leaves


def count_params(tree) -> int:
    total = 0
    for x in flatten_params(tree):
        if hasattr(x, "size"):
            total += int(x.size)
    return total


# ----------------- Build optimizers & FLOPs proxies -----------------


def build_optimizers_and_flops(cfg: RunCfg, init_params) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Build CeLO variants + conventional optimizers and assign a FLOPs proxy
    per training step, for a *specific model*.

    FLOPs proxy (very rough, but consistent):
      model_flops = N_model * model_bits
      optimizer_overhead:
        - CeLO:       N_theta * celo_bits * celo_overhead_factor
        - CeLO pruned: same but scaled by (1 - sparsity)
        - CeLO quant: N_theta * celo_quant_bits * celo_overhead_factor
        - Adam:      N_model * model_bits * adam_overhead_factor
        - Adafactor: N_model * model_bits * adafactor_overhead_factor
        - SGD:       N_model * model_bits * sgd_overhead_factor

      total_flops_per_step = model_flops + optimizer_overhead
    """
    optimizers: Dict[str, Any] = {}
    flops_per_step: Dict[str, float] = {}

    # ---- Model FLOPs ----
    n_model = count_params(init_params)
    model_flops = float(max(n_model * cfg.model_bits, 1.0))
    print(f"    [INFO] Model params: {n_model:,}")
    print(f"    [INFO] Model FLOPs proxy/step: {model_flops:,.0f} (N_model * {cfg.model_bits} bits)")

    # ---- CeLO theta FLOPs ----
    # CeLO theta size is independent of the model; loaded from ckpt
    from celo.factory import get_optimizer
    from celo.utils import load_state

    lopt_theta = get_optimizer("celo")
    theta_template = lopt_theta.init(jax.random.PRNGKey(0))
    theta_base = load_state(cfg.celo_ckpt, theta_template)
    n_theta = count_params(theta_base)
    celo_theta_flops = float(max(n_theta * cfg.celo_bits * cfg.celo_overhead_factor, 1.0))

    print(f"    [INFO] CeLO theta params: {n_theta:,}")
    print(
        f"    [INFO] CeLO FLOPs proxy/step (baseline): "
        f"{celo_theta_flops:,.0f} (N_theta * {cfg.celo_bits} bits * factor={cfg.celo_overhead_factor})"
    )

    # ---- CeLO baseline ----
    _, opt_celo = build_celo_from_ckpt(cfg.celo_ckpt, variant="baseline")
    optimizers["celo"] = opt_celo
    flops_per_step["celo"] = model_flops + celo_theta_flops

    # ---- CeLO pruned ----
    if cfg.prune_sparsity > 0.0:
        keep_ratio = 1.0 - cfg.prune_sparsity
        _, opt_pruned = build_celo_from_ckpt(
            cfg.celo_ckpt, variant="pruned", sparsity=cfg.prune_sparsity
        )
        optimizers["celo_prune"] = opt_pruned
        pruned_theta_flops = celo_theta_flops * keep_ratio
        flops_per_step["celo_prune"] = model_flops + pruned_theta_flops
        print(
            f"    [INFO] CeLO pruned: sparsity={cfg.prune_sparsity:.2f}, "
            f"keep_ratio={keep_ratio:.2f}, theta FLOPs≈{pruned_theta_flops:,.0f}"
        )

    # ---- CeLO quantized ----
    if cfg.use_quant8:
        _, opt_q8 = build_celo_from_ckpt(cfg.celo_ckpt, variant="quant8")
        optimizers["celo_q8"] = opt_q8
        q8_theta_flops = float(max(n_theta * cfg.celo_quant_bits * cfg.celo_overhead_factor, 1.0))
        flops_per_step["celo_q8"] = model_flops + q8_theta_flops
        print(
            f"    [INFO] CeLO quant8: bits={cfg.celo_quant_bits}, "
            f"theta FLOPs≈{q8_theta_flops:,.0f}"
        )

    # ---- Optional two-stage CeLO set ----
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
        optimizers["celo2"] = opt_celo2
        optimizers["celo2_prune"] = opt_prune2
        optimizers["celo2_q8"] = opt_quant2
        # For FLOPs proxy, reuse same theta size as baseline CeLO:
        flops_per_step["celo2"] = model_flops + celo_theta_flops
        flops_per_step["celo2_prune"] = model_flops + celo_theta_flops * (1.0 - cfg.prune_sparsity)
        flops_per_step["celo2_q8"] = model_flops + q8_theta_flops

    # ---- Conventional optimizers (Optax) ----
    opt_adam = OptaxAdapter(optax.adam(cfg.adam_lr, b1=cfg.adam_b1, b2=cfg.adam_b2))
    optimizers["adam"] = opt_adam
    adam_overhead = n_model * cfg.model_bits * cfg.adam_overhead_factor
    flops_per_step["adam"] = model_flops + adam_overhead
    print(
        f"    [INFO] Adam overhead FLOPs/step≈{adam_overhead:,.0f} "
        f"(factor={cfg.adam_overhead_factor})"
    )

    opt_adafactor = OptaxAdapter(optax.adafactor(learning_rate=cfg.adafactor_lr))
    optimizers["adafactor"] = opt_adafactor
    adaf_overhead = n_model * cfg.model_bits * cfg.adafactor_overhead_factor
    flops_per_step["adafactor"] = model_flops + adaf_overhead
    print(
        f"    [INFO] Adafactor overhead FLOPs/step≈{adaf_overhead:,.0f} "
        f"(factor={cfg.adafactor_overhead_factor})"
    )

    opt_sgd = OptaxAdapter(optax.sgd(learning_rate=cfg.sgd_lr, momentum=cfg.sgd_momentum))
    optimizers["sgd"] = opt_sgd
    sgd_overhead = n_model * cfg.model_bits * cfg.sgd_overhead_factor
    flops_per_step["sgd"] = model_flops + sgd_overhead
    print(
        f"    [INFO] SGD overhead FLOPs/step≈{sgd_overhead:,.0f} "
        f"(factor={cfg.sgd_overhead_factor})"
    )

    return optimizers, flops_per_step


# ------------------------ Single-dataset run -------------------------


def run_single_dataset(
    dataset_name: str,
    task_ctor: Callable[[], Any],
    cfg: RunCfg,
    csv_writer: csv.writer,
    run_idx: int,
):
    """Run all optimizers on a single dataset/task and log to CSV (with FLOPs proxy)."""

    print(f"\n=== Running dataset: {dataset_name} (run {run_idx}) ===")

    run_seed = cfg.seed + 1000 * run_idx

    base_key, task, init_params, init_state = make_task_and_init(task_ctor, run_seed)
    train_stream = task.datasets.train
    val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train

    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    # ---- Build optimizers + FLOPs proxies for THIS model ----
    opt_objs, flops_per_step = build_optimizers_and_flops(cfg, init_params)
    names = list(opt_objs.keys())

    # Independent copies of params/state per optimizer
    params0 = clone_tree(init_params)
    state0 = clone_tree(init_state)
    states: Dict[str, Any] = {}
    for name, opt in opt_objs.items():
        states[name] = opt.init(
            clone_tree(params0), model_state=clone_tree(state0), num_steps=cfg.num_steps
        )

    # Warmup to JIT-compile loss_and_grad
    warm_batch = next(train_stream)
    for idx, n in enumerate(names):
        k_warm = jax.random.fold_in(base_key, 0xC0FFEE + idx)
        opt, st = opt_objs[n], states[n]
        params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
        _ = loss_and_grad_fn(params, k_warm, warm_batch)

    # Track last train loss & cumulative FLOPs per optimizer
    last_train_loss: Dict[str, float] = {n: np.nan for n in names}
    cum_flops: Dict[str, float] = {n: 0.0 for n in names}

    main_key = base_key

    # Optionally log step=0 baseline
    val_batch0 = next(val_stream)
    for idx, n in enumerate(names):
        opt, st = opt_objs[n], states[n]
        params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
        k0 = jax.random.fold_in(base_key, 0xDEADBEEF + idx)
        val_loss0 = eval_loss_fn(params, k0, val_batch0)
        # No train loss yet; re-use val as a starting reference
        csv_writer.writerow([dataset_name, run_idx, 0, n, val_loss0, val_loss0, 0.0])

    # ---- Training loop ----
    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        for idx, n in enumerate(names):
            opt = opt_objs[n]
            st = states[n]

            k = jax.random.fold_in(main_key, step * 997 + idx)
            params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
            tr_loss, grad = loss_and_grad_fn(params, k, batch)

            st = opt.update(st, grad, loss=tr_loss)
            states[n] = st

            tr_loss_val = float(tr_loss)
            last_train_loss[n] = tr_loss_val
            cum_flops[n] += flops_per_step[n]

        if step % cfg.eval_every == 0:
            # Use one validation batch for all optimizers
            try:
                val_batch = next(val_stream)
            except StopIteration:
                val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train
                val_batch = next(val_stream)

            for idx, n in enumerate(names):
                opt = opt_objs[n]
                st = states[n]
                params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
                k_val = jax.random.fold_in(main_key, step * 4242 + idx)
                val_loss = eval_loss_fn(params, k_val, val_batch)
                csv_writer.writerow(
                    [
                        dataset_name,
                        run_idx,
                        step,
                        n,
                        last_train_loss[n],
                        val_loss,
                        cum_flops[n],
                    ]
                )

            msg = " | ".join(
                [f"{n}: tr={last_train_loss[n]:.4f}" for n in names]
            )
            print(f"[{dataset_name} step {step:05d}] {msg}")


# ----------------------------- Plotting ------------------------------

def make_flops_plots(cfg: RunCfg):
    """One FLOPs-vs-loss plot per dataset using mean ± std over runs."""

    df = pd.read_csv(cfg.csv_path)
    if df.empty:
        print("[WARN] CSV is empty; no plots will be generated.")
        return

    for dataset in sorted(df["dataset"].unique()):
        df_d = df[df["dataset"] == dataset].copy()

        plt.figure(figsize=(8, 5))
        plt.title(f"{dataset}: Training Loss vs FLOPs (mean ± std over runs)")

        for opt_name in sorted(df_d["optimizer"].unique()):
            sub_all = df_d[df_d["optimizer"] == opt_name].copy()

            # Aggregate over runs at each step
            grouped = (
                sub_all
                .groupby("step", as_index=False)
                .agg(
                    mean_flops=("cum_flops", "mean"),
                    mean_train=("train_loss", "mean"),
                    std_train=("train_loss", "std"),
                )
            )

            # Handle std==NaN when num_repeats == 1
            grouped["std_train"] = grouped["std_train"].fillna(0.0)

            x = grouped["mean_flops"].to_numpy()
            y = grouped["mean_train"].to_numpy()
            y_std = grouped["std_train"].to_numpy()

            plt.plot(x, y, label=opt_name)
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        plt.xlabel("Cumulative FLOPs proxy")
        plt.ylabel("Training loss (mean ± 1 std across runs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f"{cfg.plot_prefix}_{dataset}_train_vs_flops_mean_std.pdf"
        plt.savefig(out_path, dpi=300)
        print(f"[PLOT] Saved {out_path}")



# ------------------------------ Main --------------------------------


def main():
    cfg = CFG

    # Run all datasets and write one big CSV
    with open(cfg.csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["dataset", "run", "step", "optimizer", "train_loss", "val_loss", "cum_flops"])

        for dataset_name, task_ctor in DATASET_TASKS.items():
            for run_idx in range(cfg.num_repeats):
                run_single_dataset(dataset_name, task_ctor, cfg, writer, run_idx)

    print(f"\n[INFO] All datasets complete. CSV written to {cfg.csv_path}")
    make_flops_plots(cfg)


if __name__ == "__main__":
    main()

