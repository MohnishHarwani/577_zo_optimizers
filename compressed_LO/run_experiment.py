
#!/usr/bin/env python3
# compare_optimizers.py
import argparse, csv, time, math
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

from learned_optimization.tasks.fixed.transformer_lm import TransformerLM_LM1B_MultiRuntime_0
from compress_celo import build_celo_from_ckpt, build_celo_two_stage


@dataclass
class RunCfg:
    num_steps: int = 2_000
    eval_every: int = 100
    target_val_loss: float = 5.2
    csv_path: str = "optimizer_benchmark.csv"
    log_header_every: int = 20
    celo_ckpt: str = "./theta.state"
    celo_phase1_ckpt: str = "./theta_phase1.state"
    celo_phase2_ckpt: str = "./theta_phase2.state"
    enable_two_stage_second_set: bool = True
    prune_sparsity: float = 0.50
    use_quant8: bool = True
    adam_lr: float = 3e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adafactor_lr: float = 3e-4
    sgd_lr: float = 1e-2
    sgd_momentum: float = 0.9
    seed: int = 7


class OptaxAdapter:
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


def make_task_and_init(seed: int):
    key = jax.random.PRNGKey(seed)
    task = TransformerLM_LM1B_MultiRuntime_0()
    key, k1 = jax.random.split(key)
    params, model_state = task.init_with_state(k1)
    return key, task, params, model_state

def evaluate(task, params, val_stream, key):
    try:
        batch = next(val_stream)
    except StopIteration:
        val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train
        batch = next(val_stream)
    return float(task.loss(params, key, batch))

def fmt(x):
    return f"{x:.4f}" if isinstance(x, (float, int)) else str(x)

def banner(names):
    cols = ["step"]
    for n in names:
        cols.extend([f"{n}:tr", f"{n}:val"])
    return " | ".join(f"{c:>12}" for c in cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_steps", type=int, default=RunCfg.num_steps)
    ap.add_argument("--eval_every", type=int, default=RunCfg.eval_every)
    ap.add_argument("--target_val_loss", type=float, default=RunCfg.target_val_loss)
    ap.add_argument("--csv_path", type=str, default=RunCfg.csv_path)
    ap.add_argument("--celo_ckpt", type=str, default=RunCfg.celo_ckpt)

    ap.add_argument("--celo_phase1_ckpt", type=str, default=RunCfg.celo_phase1_ckpt,
                help="Optional path to phase-1 theta (second model).")
    ap.add_argument("--celo_phase2_ckpt", type=str, default=RunCfg.celo_phase2_ckpt,
                    help="Optional path to phase-2 theta (second model).")
    ap.add_argument("--enable_two_stage_second_set", action="store_true",
                    help="If set, also build base/pruned/quant8 for the (phase1,phase2) second model.", default=RunCfg.enable_two_stage_second_set)

    ap.add_argument("--prune_sparsity", type=float, default=RunCfg.prune_sparsity)
    ap.add_argument("--seed", type=int, default=RunCfg.seed)
    ap.add_argument("--adam_lr", type=float, default=RunCfg.adam_lr)
    ap.add_argument("--adafactor_lr", type=float, default=RunCfg.adafactor_lr)
    ap.add_argument("--sgd_lr", type=float, default=RunCfg.sgd_lr)
    ap.add_argument("--sgd_momentum", type=float, default=RunCfg.sgd_momentum)
    args = ap.parse_args()

    cfg = RunCfg(
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        target_val_loss=args.target_val_loss,
        csv_path=args.csv_path,
        celo_ckpt=args.celo_ckpt,
        prune_sparsity=args.prune_sparsity,
        adam_lr=args.adam_lr,
        adafactor_lr=args.adafactor_lr,
        sgd_lr=args.sgd_lr,
        sgd_momentum=args.sgd_momentum,
        seed=args.seed,
    )

    # Task + init
    base_key, task, init_params, init_state = make_task_and_init(cfg.seed)
    train_stream = task.datasets.train
    val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train

    # ---- step functions that CLOSE OVER `task` (no task arg to jit) ----
    def make_step_fns(task):
        def to_jnp_tree(x):
            # Convert anything array-like to jnp.array; leave scalars/None alone.
            return jnp.asarray(x) if hasattr(x, "dtype") else x

        def loss_fn(params, key, batch):
            # Ensure batch is purely JAX arrays (dicts/lists/np arrays → jnp)
            batch = jax.tree.map(to_jnp_tree, batch)
            return task.loss(params, key, batch)

        # Remove donate_argnums to avoid PJRT layout assertion on some setups.
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        # Keep eval simple (host eval is fine and avoids more compiled entrypoints)
        def eval_loss(params, key, batch):
            batch = jax.tree.map(to_jnp_tree, batch)
            return float(task.loss(params, key, batch))
        return loss_and_grad, eval_loss

    loss_and_grad_fn, eval_loss_fn = make_step_fns(task)

    # === Build optimizers ===
    opt_objs: Dict[str, Any] = {}
    _, opt_celo   = build_celo_from_ckpt(cfg.celo_ckpt, variant="baseline")
    _, opt_pruned = build_celo_from_ckpt(cfg.celo_ckpt, variant="pruned", sparsity=cfg.prune_sparsity)
    _, opt_quant8 = build_celo_from_ckpt(cfg.celo_ckpt, variant="quant8")
    opt_adam      = OptaxAdapter(optax.adam(cfg.adam_lr, b1=cfg.adam_b1, b2=cfg.adam_b2))
    opt_adafactor = OptaxAdapter(optax.adafactor(learning_rate=cfg.adafactor_lr))
    opt_sgd       = OptaxAdapter(optax.sgd(learning_rate=cfg.sgd_lr, momentum=cfg.sgd_momentum))

    opt_objs["celo"]       = opt_celo
    opt_objs["celo_prune"] = opt_pruned
    opt_objs["celo_q8"]    = opt_quant8
    opt_objs["adam"]       = opt_adam
    opt_objs["adafactor"]  = opt_adafactor
    opt_objs["sgd"]        = opt_sgd
        
    if args.enable_two_stage_second_set and args.celo_phase1_ckpt and args.celo_phase2_ckpt:
        _, opt_celo2   = build_celo_two_stage(args.celo_phase1_ckpt, args.celo_phase2_ckpt,
                                              variant="baseline")
        _, opt_prune2  = build_celo_two_stage(args.celo_phase1_ckpt, args.celo_phase2_ckpt,
                                              variant="pruned", sparsity=cfg.prune_sparsity)
        _, opt_quant2  = build_celo_two_stage(args.celo_phase1_ckpt, args.celo_phase2_ckpt,
                                              variant="quant8")
        # distinct names so CSV/plots separate them clearly
        opt_objs["celo2"]       = opt_celo2
        opt_objs["celo2_prune"] = opt_prune2
        opt_objs["celo2_q8"]    = opt_quant2

    # Independent copies of params/state
    def clone(x): return jtu.tree_map(lambda a: a, x)
    states: Dict[str, Any] = {}
    params0 = clone(init_params)
    state0  = clone(init_state)
    for name, opt in opt_objs.items():
        states[name] = opt.init(clone(params0), model_state=clone(state0), num_steps=cfg.num_steps)

    names = list(opt_objs.keys())
    best_global_time: Dict[str, float] = {n: math.inf for n in names}
    accum_self_time: Dict[str, float]  = {n: 0.0 for n in names}
    start_global = time.perf_counter()

    # ---------- JIT warmup (compile) so timing is cleaner ----------
    warm_batch = next(train_stream)
    for idx, n in enumerate(names):
        k_warm = jax.random.fold_in(base_key, 0xC0FFEE + idx)
        opt, st = opt_objs[n], states[n]
        params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
        _ = loss_and_grad_fn(params, k_warm, warm_batch)

    # CSV
    with open(cfg.csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        header = ["step"]
        for n in names:
            header += [f"{n}_train", f"{n}_val", f"{n}_accum_sec", f"{n}_hit_sec"]
        writer.writerow(header)

        print(banner(names))
        print("-" * (len(header) * 14))

        # Training
        main_key = base_key
        for step in range(1, cfg.num_steps + 1):
            try:
                batch = next(train_stream)
            except StopIteration:
                train_stream = task.datasets.train
                batch = next(train_stream)

            row = [step]
            for idx, n in enumerate(names):
                k = jax.random.fold_in(main_key, step * 997 + idx)
                opt = opt_objs[n]
                st  = states[n]
                params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)

                t0 = time.perf_counter()
                tr_loss, grad = loss_and_grad_fn(params, k, batch)   # <-- FIX: no `task` arg
                st = opt.update(st, grad, loss=tr_loss)
                t1 = time.perf_counter()
                accum_self_time[n] += (t1 - t0)
                states[n] = st

                if step % cfg.eval_every == 0:
                    val_key = jax.random.fold_in(main_key, 100_000 + step * 13 + idx)
                    val_params = st[0] if isinstance(opt, OptaxAdapter) else opt.get_params(st)
                    # v = evaluate(task, val_params, val_stream, val_key)
                    try:
                        val_batch = next(val_stream)
                    except StopIteration:
                        val_stream = task.datasets.valid if hasattr(task.datasets, "valid") else task.datasets.train
                        val_batch = next(val_stream)
                    v = eval_loss_fn(val_params, val_key, val_batch)
                else:
                    v = float("nan")

                row += [float(tr_loss), v, accum_self_time[n],
                        (time.perf_counter() - start_global) if (v <= cfg.target_val_loss) and (best_global_time[n] == math.inf) else math.inf]

                if (not math.isnan(v)) and v <= cfg.target_val_loss and best_global_time[n] == math.inf:
                    best_global_time[n] = time.perf_counter() - start_global

            writer.writerow(row)

            if step % cfg.eval_every == 0:
                parts = [f"{step:>12d}"]
                for i, n in enumerate(names):
                    tr = row[1 + i*4 + 0]
                    vl = row[1 + i*4 + 1]
                    parts.append(f"{fmt(tr):>12}")
                    parts.append(f"{fmt(vl):>12}")
                print(" | ".join(parts))

            if all(t < math.inf for t in best_global_time.values()):
                break

        print("\n=== Time to target val loss (global wall-clock) ===")
        for n in names:
            t = best_global_time[n]
            print(f"{n:>12}: {t:.2f} s" if t < math.inf else f"{n:>12}: (did not hit ≤ {cfg.target_val_loss})")

        print("\n=== Accumulated exclusive step time (approx) ===")
        for n in names:
            print(f"{n:>12}: {accum_self_time[n]:.2f} s")

        print(f"\nCSV written to: {cfg.csv_path}")


if __name__ == "__main__":
    main()
