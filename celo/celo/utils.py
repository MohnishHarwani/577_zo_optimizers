"""
Collection of all the util methods used in this repo.
"""

import functools
from collections import OrderedDict
from typing import TypeVar

import jax
import jax.numpy as jnp
import learned_optimization
from absl import logging
from flax import serialization
from learned_optimization.optimizers import base as opt_base

T = TypeVar("T")


def pytree_to_ordered_dict(pytree):
    # since jax tree flatten order is deterministic, we return ordered dict
    key_tree = learned_optimization.tree_utils.map_named(lambda k, v: k, pytree)
    keys, _ = jax.tree_util.tree_flatten(key_tree)
    vals, _ = jax.tree_util.tree_flatten(pytree)
    return OrderedDict(zip(keys, vals))


def load_from_pretrained(input_tree, pretrained_tree):
    input_dict = pytree_to_ordered_dict(input_tree)
    pretrained_dict = pytree_to_ordered_dict(pretrained_tree)
    updated_vals = []
    not_found = []
    # copy params from pretrained_dict which exist in param_dict
    for k, v in input_dict.items():
        val = None
        if k in pretrained_dict:
            val = pretrained_dict[k]
        else:
            val = v
            not_found.append(k)
        updated_vals.append(val)
    if len(not_found) > 0:
        logging.info(f"Params not found in checkpoint: {not_found}")
    struct = jax.tree_util.tree_structure(input_tree)
    output_tree = struct.unflatten(updated_vals)
    return output_tree


def load_state(path: str, state: T) -> T:
    """Load a pytree state directly from a file.

    Args:
      path: path to load pytree state from.
      state: pytree whose structure should match that of the stucture saved in the
        path. The values of this pytree are not used.

    Returns:
      The restored pytree matching the pytree structure of state.
    """
    with open(path, "rb") as fp:
        state_new = serialization.from_bytes(state, fp.read())
    tree = jax.tree_util.tree_structure(state)
    leaves_new = jax.tree_util.tree_leaves(state_new)
    return jax.tree_util.tree_unflatten(tree, leaves_new)


def init_lopt_from_ckpt(lopt, path, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    theta = lopt.init(key)
    pretrained_params = load_state(path, theta)
    opt = lopt.opt_fn(pretrained_params)
    return opt


@functools.lru_cache(None)
def cached_jit(fn, *args, **kwargs):
    return jax.jit(fn, *args, **kwargs)


class WeightDecayWrapper(opt_base.Optimizer):
    def __init__(self, opt, weight_decay=0.0, add_to_loss=True):
        super().__init__()
        self.opt = opt
        self.weight_decay = weight_decay
        self.add_to_loss = add_to_loss

    def get_params(self, opt_state):
        self.opt.get_params(opt_state)
        return self.opt.get_params(opt_state)

    def set_params(self, state, params):
        return self.opt.set_params(state, params)

    def get_state(self, opt_state):
        return self.opt.get_state(opt_state)

    def init(self, params, model_state=None, **kwargs):
        return self.opt.init(params, model_state=model_state, **kwargs)

    def update(self, opt_state, grads, ps, loss=None, model_state=None, **kwargs):
        if self.add_to_loss:
            l2 = [jnp.sum(p**2) for p in jax.tree_util.tree_leaves(ps)]
            loss = loss + sum([x * self.weight_decay for x in l2])

        grad_l2 = jax.tree_util.tree_map(lambda p: self.weight_decay * p, ps)
        grads = jax.tree_util.tree_map(lambda g, g_l2: g + g_l2, grads, grad_l2)

        return self.opt.update(opt_state, grads, ps, loss=loss, model_state=model_state, **kwargs)
