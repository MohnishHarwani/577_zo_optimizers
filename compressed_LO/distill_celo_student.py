#!/usr/bin/env python3
# distill_celo_student.py
import os, pickle, math, functools
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax, jax.numpy as jnp
from jax import tree_util as jtu
import optax
import haiku as hk

from learned_optimization.tasks.fixed.transformer_lm import (
    TransformerLM_LM1B_MultiRuntime_0,
)
from compress_celo import build_celo_from_ckpt  # you already have this


# ------------------ Tiny per-value MLP ------------------
def _feat_map(x):
    ax = jnp.abs(x)
    return jnp.stack([x, jnp.sign(x), jnp.log(ax + 1e-8)], axis=-1)


class PerValueMLP(hk.Module):
    def __init__(self, hidden=32, name=None):
        super().__init__(name=name)
        self.h = hidden

    def __call__(self, x):
        feats = _feat_map(x)
        flat = feats.reshape(-1, feats.shape[-1])
        mlp = hk.nets.MLP([self.h, self.h, 1], activate_final=False)
        out = mlp(flat).reshape(x.shape)
        return out


def student_predict_update(grads):
    net = PerValueMLP(hidden=32)
    return jtu.tree_map(lambda g: net(g), grads)


student_apply = hk.without_apply_rng(hk.transform(student_predict_update))


# ------------------ Utils ------------------
def tree_sub(a, b):
    return jtu.tree_map(lambda x, y: x - y, a, b)


def tree_add(a, b):
    return jtu.tree_map(lambda x, y: x + y, a, b)


def tree_l2(a):
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(a)]) + 1e-9)


def tree_cos(a, b):
    num = sum([jnp.sum(x * y) for x, y in zip(jtu.tree_leaves(a), jtu.tree_leaves(b))])
    den = tree_l2(a) * tree_l2(b) + 1e-9
    return num / den


def to_jnp_tree(batch):
    return jax.tree.map(lambda x: jnp.asarray(x) if hasattr(x, "dtype") else x, batch)


# ------------------ KD config ------------------
@dataclass
class KDConfig:
    # interpret `steps` as steps per epoch now
    steps: int = 2_000
    epochs: int = 2_000          # NEW: number of epochs
    lr: float = 1e-3
    eval_every: int = 100    # based on *global* step
    ckpt_path: str = "student_kd.pkl"
    celo_ckpt: str = "./theta.state"


def make_task(seed=7):
    key = jax.random.PRNGKey(seed)
    task = TransformerLM_LM1B_MultiRuntime_0()
    key, k1 = jax.random.split(key)
    params, model_state = task.init_with_state(k1)
    return key, task, params, model_state


def kd_train(cfg: KDConfig):
    # Task + streams
    base_key, task, params0, state0 = make_task()
    train_stream = task.datasets.train

    # Teacher CeLO (frozen)
    _, opt_teacher = build_celo_from_ckpt(cfg.celo_ckpt, variant="baseline")
    t_state = opt_teacher.init(params0, model_state=state0, num_steps=cfg.steps * cfg.epochs)

    # Init student
    dummy_grads = jtu.tree_map(jnp.zeros_like, params0)
    s_params = student_apply.init(jax.random.PRNGKey(0), dummy_grads)
    s_opt = optax.adam(cfg.lr)
    s_opt_state = s_opt.init(s_params)

    # loss+grad that closes over `task`
    def loss_fn(params, key, batch):
        batch = to_jnp_tree(batch)
        return task.loss(params, key, batch)

    task_loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def kd_step(s_params, s_opt_state, t_state, params, key, batch):
        # teacher loss and grads
        tr_loss, grads = task_loss_and_grad(params, key, batch)
        # teacher one-step update (local state; we don't feed updated params back into loop)
        new_t_state = opt_teacher.update(t_state, grads, loss=tr_loss)
        params_next_T = opt_teacher.get_params(new_t_state)
        delta_T = tree_sub(params_next_T, params)

        def loss_fn_student(p):
            delta_S = student_apply.apply(p, grads)
            l2 = tree_l2(tree_sub(delta_S, delta_T))
            cos = 1.0 - tree_cos(delta_S, delta_T)
            reg = tree_l2(delta_S) * 1e-4
            return l2 + 0.25 * cos + reg, (l2, cos)

        (loss, (l2, cos)), grads_theta = jax.value_and_grad(
            loss_fn_student, has_aux=True
        )(s_params)
        updates, new_opt_state = s_opt.update(grads_theta, s_opt_state, s_params)
        new_s_params = optax.apply_updates(s_params, updates)
        # keep teacher state fixed to make the target stationary
        return new_s_params, new_opt_state, t_state, loss, l2, cos

    params = params0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n=== KD Epoch {epoch}/{cfg.epochs} ===")
        # (re)start train stream each epoch
        train_stream = task.datasets.train

        for step in range(1, cfg.steps + 1):
            try:
                batch = next(train_stream)
            except StopIteration:
                train_stream = task.datasets.train
                batch = next(train_stream)

            global_step += 1
            key = jax.random.fold_in(base_key, global_step)

            s_params, s_opt_state, t_state, loss, l2, cos = kd_step(
                s_params, s_opt_state, t_state, params, key, batch
            )

            if global_step % cfg.eval_every == 0:
                print(
                    f"[KD epoch={epoch} step={step} global_step={global_step}] "
                    f"distill_loss={float(loss):.4f}  L2={float(l2):.4f}  1-cos={float(cos):.4f}"
                )

    with open(cfg.ckpt_path, "wb") as f:
        pickle.dump({"student_params": jax.device_get(s_params)}, f)
    print(f"Saved student to {cfg.ckpt_path}")


# ------------- Adapter so you can benchmark the student -------------
class StudentAdapter:
    """Mimics init/get_params/update like your other optimizers."""

    def __init__(self, student_params):
        self.p = student_params

    def init(self, params, model_state=None, num_steps=0):
        return (params, None)

    def get_params(self, state):
        params, _ = state
        return params

    def update(self, state, grad, loss=None):
        params, _ = state
        delta = student_apply.apply(self.p, grad)
        new_params = tree_add(params, delta)
        return (new_params, None)


if __name__ == "__main__":
    kd_train(KDConfig())

