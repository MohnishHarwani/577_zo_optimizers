"""
Extends task augmentation implementation in learned optimization library
to support augmentation at different levels (global, tensor, parameter, none)
via a sampling list as in VeLO. Also, adds small image MLP meta-training tasks
from VeLO.

Adapted from:
- https://github.com/google/learned_optimization/blob/2892faec4d5b24a419b6892056af8ee6f7920309/learned_optimization/tasks/task_augmentation.py
- https://github.com/google/learned_optimization/blob/2892faec4d5b24a419b6892056af8ee6f7920309/learned_optimization/research/general_lopt/tasks/fast_mlp_diff_data.py
"""

from typing import Any, List, Mapping, Tuple, Union

import gin
import jax
import jax.numpy as jnp
import numpy as onp
from learned_optimization import summary
from learned_optimization.tasks import base
from learned_optimization.tasks.base import Batch, ModelState, Params
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.parametric import cfgobject

PRNGKey = jnp.ndarray
PyTree = Any

LogFeat = cfgobject.LogFeature


def cifar_task(image_size=8, bw=True):
    inner_bs = 64
    datasets = image.cifar10_datasets(
        batch_size=inner_bs,
        image_size=(image_size, image_size),
        convert_to_black_and_white=bw,
    )
    return image_mlp._MLPImageTask(datasets, [32])


def fashion_mnist_task(image_size=8):
    inner_bs = 64
    datasets = image.fashion_mnist_datasets(
        batch_size=inner_bs, image_size=(image_size, image_size)
    )
    return image_mlp._MLPImageTask(datasets, [32])


def mnist_task(image_size=8):
    inner_bs = 64
    datasets = image.mnist_datasets(batch_size=inner_bs, image_size=(image_size, image_size))
    return image_mlp._MLPImageTask(datasets, [32])


def svhn_task(image_size=8, bw=True):
    inner_bs = 64
    datasets = image.svhn_cropped_datasets(
        batch_size=inner_bs,
        image_size=(image_size, image_size),
        convert_to_black_and_white=bw,
    )
    return image_mlp._MLPImageTask(datasets, [32])


class ReparamWeights(base.Task):
    """Reparameterize weights of target task by the param_scale.

    If the underlying loss is f(x;w) = w@x, this function transforms the loss to
    be f(x;w) = (w*param_scale)@x and changes the initial weights to be:
    w=w0/param_scale where w0 is the provided Task's init.

    This reparameterization does NOT change the underlying function, but does
    change the learning dynamics of the problem greatly as the underlying params
    will be more, or less sensitive.
    """

    def __init__(self, task: base.Task, param_scale: Union[Params, float]):
        super().__init__()
        self.task = task
        self.normalizer = task.normalizer
        self.datasets = task.datasets
        self._param_scale = param_scale

    def _match_param_scale_to_pytree(self, params: Params) -> Params:
        if isinstance(self._param_scale, (jnp.ndarray, onp.ndarray, float, int, onp.float32)):
            return jax.tree_util.tree_map(lambda x: self._param_scale, params)
        else:
            tree = jax.tree_util.tree_structure(params)
            tree_scale = jax.tree_util.tree_structure(self._param_scale)
            assert tree == tree_scale, f"Structures: {tree} AND {tree_scale}"
            return self._param_scale

    def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
        params, state = self.task.init_with_state(key)
        scales = self._match_param_scale_to_pytree(params)
        params = jax.tree_util.tree_map(lambda x, scale: x / scale, params, scales)
        return params, state

    def init(self, key: PRNGKey) -> Params:
        params, _ = self.init_with_state(key)
        return params

    def loss_with_state_and_aux(
        self, params: Params, state: ModelState, key: PRNGKey, data: Batch
    ) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
        scales = self._match_param_scale_to_pytree(params)
        params = jax.tree_util.tree_map(lambda x, scale: x * scale, params, scales)
        return self.task.loss_with_state_and_aux(params, state, key, data)

    def loss(
        self, params: Params, key: PRNGKey, data: Batch
    ) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
        loss, _, _ = self.loss_with_state_and_aux(params, None, key, data)
        return loss

    def loss_with_state(
        self, params: Any, state: Any, key: jnp.ndarray, data: Any
    ) -> Tuple[jnp.ndarray, Any]:
        loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
        return loss, state


@gin.configurable
class ReparamWeightsListFamily(base.TaskFamily):
    """Reparam the weights of a TaskFamily from given lists."""

    def __init__(
        self,
        task_family: base.TaskFamily,
        levels: List[str] = ["global", "tensor", "parameter", "none"],
        param_scale_ranges: List[Tuple[float, float]] = [
            (0.001, 1000.0),
            (0.01, 100.0),
            (0.1, 10.0),
        ],
    ):
        super().__init__()
        for level in levels:
            assert level in ["global", "tensor", "parameter", "none"]
        level2idx = {"global": 0, "tensor": 1, "parameter": 2, "none": 3}
        self._levels = jnp.array([level2idx[k] for k in levels])
        self.task_family = task_family
        self._param_scale_ranges = onp.array(param_scale_ranges)
        self.datasets = task_family.datasets
        self._name = f"ReparamWeightsList_{task_family.name}"

    def _single_random(self, p, key, scale_range):
        min_val, max_val = scale_range
        param_scale = jax.random.uniform(key, [], minval=jnp.log(min_val), maxval=jnp.log(max_val))
        return jnp.exp(param_scale) * jnp.ones(p.shape)

    def _single_fixed(self, p, key, fixed_val):
        return fixed_val * jnp.ones(p.shape)

    def task_fn(self, cfg: cfgobject.CFGNamed) -> base.Task:
        # check if sampled config has parameter level or globa/tensor one
        _level = cfg.values["level"]
        _param_scale_range = cfg.values["param_scale_range"]
        key2 = cfg.values["key"]
        sub_config = cfg.values["sub_cfg"]

        def pscale_level0(key):  # global
            key2 = key
            key1, key2 = jax.random.split(key2)
            param_scale = self._single_random(jnp.array(1.0), key1, _param_scale_range)
            param_scale_val = LogFeat(param_scale).value
            abstract_params, _ = jax.eval_shape(
                lambda key: self.task_family.sample_task(key).init_with_state(key),
                jax.random.PRNGKey(0),
            )
            leaves, tree = jax.tree_util.tree_flatten(abstract_params)
            keys = jax.tree_util.tree_unflatten(tree, jax.random.split(key2, len(leaves)))
            scales = jax.tree_util.tree_unflatten(tree, [param_scale_val] * len(leaves))
            param_scale = jax.tree_util.tree_map(self._single_fixed, abstract_params, keys, scales)
            return param_scale

        def pscale_level1(key):  # tensor
            key2 = key
            abstract_params, _ = jax.eval_shape(
                lambda key: self.task_family.sample_task(key).init_with_state(key),
                jax.random.PRNGKey(0),
            )
            leaves, tree = jax.tree_util.tree_flatten(abstract_params)
            keys = jax.tree_util.tree_unflatten(tree, jax.random.split(key2, len(leaves)))
            scales = jax.tree_util.tree_unflatten(tree, [_param_scale_range] * len(leaves))
            param_scale = jax.tree_util.tree_map(self._single_random, abstract_params, keys, scales)
            return param_scale

        def pscale_level2(key):  # param-level
            key2 = key
            abstract_params, _ = jax.eval_shape(
                lambda key: self.task_family.sample_task(key).init_with_state(key),
                jax.random.PRNGKey(0),
            )
            leaves, tree = jax.tree_util.tree_flatten(abstract_params)
            keys = jax.tree_util.tree_unflatten(tree, jax.random.split(key2, len(leaves)))

            def single(p, key):
                min_val, max_val = _param_scale_range
                param_scale = jax.random.uniform(
                    key, p.shape, minval=jnp.log(min_val), maxval=jnp.log(max_val)
                )
                return jnp.exp(param_scale)

            param_scale = jax.tree_util.tree_map(single, abstract_params, keys)
            return param_scale

        def pscale_level3(key):  # none
            key2 = key
            param_scale_val = 1.0
            abstract_params, _ = jax.eval_shape(
                lambda key: self.task_family.sample_task(key).init_with_state(key),
                jax.random.PRNGKey(0),
            )
            leaves, tree = jax.tree_util.tree_flatten(abstract_params)
            keys = jax.tree_util.tree_unflatten(tree, jax.random.split(key2, len(leaves)))
            scales = jax.tree_util.tree_unflatten(tree, [param_scale_val] * len(leaves))
            param_scale = jax.tree_util.tree_map(self._single_fixed, abstract_params, keys, scales)
            return param_scale

        task = self.task_family.task_fn(sub_config)
        param_scale = jax.lax.switch(
            _level, [pscale_level0, pscale_level1, pscale_level2, pscale_level3], key2
        )
        return ReparamWeights(task, param_scale)

    def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
        key1, key2 = jax.random.split(key)
        sub_config = self.task_family.sample(key1)

        # sample level and param scale range
        key1, key2 = jax.random.split(key2)
        _level_idx = jax.random.choice(key1, self._levels)
        idx = onp.random.randint(len(self._param_scale_ranges))
        _param_scale_range = jnp.array(self._param_scale_ranges[idx])
        summary.summary("aug_reparam/sampled_level", _level_idx, aggregation="sample")
        summary.summary(
            "aug_reparam/sampled_range_min", _param_scale_range[0], aggregation="sample"
        )
        return cfgobject.CFGNamed(
            "ReparamWeightsFamily",
            {
                "sub_cfg": sub_config,
                "param_scale_range": _param_scale_range,
                "key": key2,
                "level": _level_idx,
            },
        )
