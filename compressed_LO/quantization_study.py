#!/usr/bin/env python3
"""
CeLO-only quantization sweep (Transformer only, x-axis = steps).

Fixes fp8 dtype-mixing errors by using FAKE quantization for fp8:
  theta_q = (theta cast to float8) cast back to float32
so Haiku dot never sees float8 weights.

Outputs:
- CSV: celo_quant_transformer_steps.csv
- Plot (PDF): celo_quant_transformer_steps_transformer_mean_std.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from learned_optimization.tasks.fixed.transformer_lm import TransformerLM_LM1B_MultiRuntime_0

# CeLO
from celo.factory import get_optimizer
from celo.utils import load_state


# --------------------------- Config ---------------------------------


@dataclass
class RunCfg:
    # Training
    num_steps: int = 2_000
    eval_every: int = 1
    seed: int = 7
    num_repeats: int = 5

    # CeLO checkpoint (theta.state)
    celo_ckpt: str = "./models/theta.state"

    # Quant sweep (order controls legend ordering)
    quant_schemes: Tuple[str, ...] = (
        "fp32",
        "bf16",
        "fp16",
        "fp8_e4m3fn",
        "fp8_e5m2",
        "int8",
        "int4",
    )

    # Fake-quant eps for int
    int_quant_eps: float = 1e-8

    # Output
    csv_path: str = "celo_quant_transformer_steps.csv"
    plot_path: str = "celo_quant_transformer_steps_transformer_mean_std.pdf"


CFG = RunCfg()


# ----------------------- Task init / loss fns ------------------------


def make_task_and_init(task_ctor: Callable[[], Any], seed: int):
    key = jax.random.PRNGKey(seed)
    task = task_ctor()
    key, k1 = jax.random.split(key)
    params, model_state = task.init_with_state(k1)
    return key, task, params, model_state


def make_step_fns(task):
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


# ----------------------- Quantization utils -------------------------


def _maybe_get_jax_dtype(attr: str):
    return getattr(jnp, attr, None)


def quantize_array_fake_int(x: jnp.ndarray, bits: int, eps: float) -> jnp.ndarray:
    """Symmetric per-tensor fake quantization: quantize -> dequantize, returned float32."""
    if not hasattr(x, "dtype") or x.dtype == jnp.bool_:
        return x
    x_f = x.astype(jnp.float32)
    qmax = (2 ** (bits - 1)) - 1
    amax = jnp.max(jnp.abs(x_f))
    scale = jnp.maximum(amax / jnp.maximum(qmax, 1), eps)
    q = jnp.clip(jnp.round(x_f / scale), -qmax, qmax)
    return (q * scale).astype(jnp.float32)


def quantize_theta(theta, scheme: str, eps: float):
    """
    Quantize CeLO theta pytree.

    IMPORTANT: fp8 schemes are done as FAKE quant:
        theta_fp8 = (cast to float8) then cast back to float32
      to avoid Haiku dot(float32, float8) dtype-promotion errors.
    """
    s = scheme.lower()

    if s in ("fp32", "float32"):
        return jtu.tree_map(lambda a: a.astype(jnp.float32) if hasattr(a, "dtype") else a, theta)

    if s in ("bf16", "bfloat16"):
        # bf16 is generally safe in dot on modern JAX, but to be robust you can also "fake" it.
        return jtu.tree_map(lambda a: a.astype(jnp.bfloat16) if hasattr(a, "dtype") else a, theta)

    if s in ("fp16", "float16"):
        return jtu.tree_map(lambda a: a.astype(jnp.float16) if hasattr(a, "dtype") else a, theta)

    if s in ("fp8_e4m3fn", "fp8e4m3fn"):
        dt = _maybe_get_jax_dtype("float8_e4m3fn")
        if dt is None:
            raise RuntimeError("Your JAX build does not expose jnp.float8_e4m3fn.")
        # FAKE fp8: cast->cast back
        return jtu.tree_map(
            lambda a: a.astype(dt).astype(jnp.float32) if hasattr(a, "dtype") else a,
            theta,
        )

    if s in ("fp8_e5m2", "fp8e5m2"):
        dt = _maybe_get_jax_dtype("float8_e5m2")
        if dt is None:
            raise RuntimeError("Your JAX build does not expose jnp.float8_e5m2.")
        # FAKE fp8: cast->cast back
        return jtu.tree_map(
            lambda a: a.astype(dt).astype(jnp.float32) if hasattr(a, "dtype") else a,
            theta,
        )

    if s == "int8":
        return jtu.tree_map(
            lambda a: quantize_array_fake_int(a, bits=8, eps=eps) if hasattr(a, "dtype") else a,
            theta,
        )

    if s == "int4":
        return jtu.tree_map(
            lambda a: quantize_array_fake_int(a, bits=4, eps=eps) if hasattr(a, "dtype") else a,
            theta,
        )

    raise ValueError(f"Unknown quant scheme: {scheme}")


# --------------------- Build CeLO variants ---------------------------


def build_celo_variants(cfg: RunCfg):
    lopt = get_optimizer("celo")
    theta_template = lopt.init(jax.random.PRNGKey(0))
    theta_fp32 = load_state(cfg.celo_ckpt, theta_template)

    opts: Dict[str, Any] = {}
    for scheme in cfg.quant_schemes:
        name = f"celo_{scheme}"
        try:
            theta_q = quantize_theta(theta_fp32, scheme=scheme, eps=cfg.int_quant_eps)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        opts[name] = lopt.opt_fn(theta_q)

    if not opts:
        raise RuntimeError("No CeLO variants constructed. Check ckpt path / dtype availability.")
    return opts


# ------------------------ Single run ---------------------------------


def run_one_repeat(cfg: RunCfg, run_idx: int, csv_writer: csv.writer):
    dataset_name = "TransformerLM_LM1B_MultiRuntime_0"
    task_ctor = TransformerLM_LM1B_MultiRuntime_0

    print(f"\n=== {dataset_name} | repeat {run_idx} ===")
    run_seed = cfg.seed + 1000 * run_idx
    base_key, task, init_params, init_state = make_task_and_init(task_ctor, run_seed)

    train_stream = task.datasets.train
    val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train

    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    opt_objs = build_celo_variants(cfg)
    names = list(opt_objs.keys())

    # per-variant state
    params0 = clone_tree(init_params)
    state0 = clone_tree(init_state)
    states: Dict[str, Any] = {}
    for name, opt in opt_objs.items():
        states[name] = opt.init(
            clone_tree(params0),
            model_state=clone_tree(state0),
            num_steps=cfg.num_steps,
        )

    # warmup compile
    warm_batch = next(train_stream)
    for i, n in enumerate(names):
        k = jax.random.fold_in(base_key, 0xC0FFEE + i)
        params = opt_objs[n].get_params(states[n])
        _ = loss_and_grad_fn(params, k, warm_batch)

    # step 0 eval
    val_batch0 = next(val_stream)
    for i, n in enumerate(names):
        params = opt_objs[n].get_params(states[n])
        k0 = jax.random.fold_in(base_key, 0xBEEF + i)
        v0 = eval_loss_fn(params, k0, val_batch0)
        csv_writer.writerow([dataset_name, run_idx, 0, n, np.nan, v0])

    main_key = base_key

    for step in range(1, cfg.num_steps + 1):
        try:
            batch = next(train_stream)
        except StopIteration:
            train_stream = task.datasets.train
            batch = next(train_stream)

        # train step
        train_losses: Dict[str, float] = {}
        for i, n in enumerate(names):
            opt = opt_objs[n]
            st = states[n]
            params = opt.get_params(st)

            k = jax.random.fold_in(main_key, step * 997 + i)
            tr_loss, grad = loss_and_grad_fn(params, k, batch)

            st = opt.update(st, grad, loss=tr_loss)
            states[n] = st
            train_losses[n] = float(tr_loss)

        # periodic eval + log
        if step % cfg.eval_every == 0:
            try:
                val_batch = next(val_stream)
            except StopIteration:
                val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train
                val_batch = next(val_stream)

            for i, n in enumerate(names):
                opt = opt_objs[n]
                st = states[n]
                params = opt.get_params(st)
                k_val = jax.random.fold_in(main_key, step * 4242 + i)
                v = eval_loss_fn(params, k_val, val_batch)
                csv_writer.writerow([dataset_name, run_idx, step, n, train_losses[n], v])

            msg = " | ".join([f"{n}: tr={train_losses[n]:.4f}" for n in names])
            print(f"[step {step:05d}] {msg}")


# ----------------------------- Plotting ------------------------------


def make_plot(cfg: RunCfg):
    df = pd.read_csv(cfg.csv_path)
    if df.empty:
        print("[WARN] CSV empty; no plot.")
        return

    df = df[df["step"] > 0].copy()  # ignore step 0 (train_loss nan)
    plt.figure(figsize=(8, 5))
    plt.title("TransformerLM_LM1B: CeLO quant sweep (mean ± std)")

    for opt_name in sorted(df["optimizer"].unique()):
        sub = df[df["optimizer"] == opt_name].copy()
        g = (
            sub.groupby("step", as_index=False)
            .agg(mean_train=("train_loss", "mean"), std_train=("train_loss", "std"))
        )
        g["std_train"] = g["std_train"].fillna(0.0)

        x = g["step"].to_numpy()
        y = g["mean_train"].to_numpy()
        ystd = g["std_train"].to_numpy()

        plt.plot(x, y, label=opt_name)
        plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Training loss (mean ± 1 std across repeats)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=300)
    print(f"[PLOT] Saved {cfg.plot_path}")


# ------------------------------ Main --------------------------------


def main():
    cfg = CFG
    with open(cfg.csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["dataset", "run", "step", "optimizer", "train_loss", "val_loss"])

        for run_idx in range(cfg.num_repeats):
            run_one_repeat(cfg, run_idx, writer)

    print(f"\n[INFO] Done. CSV written to {cfg.csv_path}")
    make_plot(cfg)


if __name__ == "__main__":
    main()

