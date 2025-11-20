# coding=utf-8
"""
Mnemo-CeLo: CeLo-style learned optimizer with Mnemosyne-inspired
topological and temporal encoders.

This implements a LearnedOptimizer with the same interface pattern as
celo.optimizers.celo.Celo, but replaces the internal "rnn + MLP update"
with:

  - a **topological encoder**: multi-head self-attention over per-parameter
    feature tokens (one token per leaf in the parameter PyTree);
  - a **temporal encoder**: simple recurrent (tanh) state per parameter
    that is updated every step.

The output is a per-parameter scalar step-size multiplier that scales
the raw gradient.

"""

from typing import Any, Dict, Tuple

import chex
import flax
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base


# ----------------------------------------------------------------------
# Simple loss buffer: multi-timescale EMA with features
# ----------------------------------------------------------------------


class BufferLossAccumulators:
    """
    Rolling accumulator for loss values across multiple timescales.
    """

    def __init__(self, num_scales: int = 8):
        self.num_scales = num_scales

    def init(self, num_steps: int) -> Dict[str, jnp.ndarray]:
        # Log-spaced half-lives between ~10 and num_steps
        num_steps = jnp.maximum(jnp.asarray(num_steps, dtype=jnp.float32), 10.0)
        halflife = jnp.logspace(1.0, jnp.log10(num_steps), self.num_scales)
        decays = jnp.exp(-1.0 / halflife)
        zeros = jnp.zeros((self.num_scales,), dtype=jnp.float32)
        big = jnp.full((self.num_scales,), 1e9, dtype=jnp.float32)

        return {
            "means": zeros,
            "iteration": jnp.asarray(0, dtype=jnp.int32),
            "running_min": big,
            "decays": decays,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, state: Dict[str, jnp.ndarray], loss: jnp.ndarray):
        """Update the rolling stats with a new loss value."""
        jdecays = state["decays"]
        loss = jnp.asarray(loss, dtype=jnp.float32)

        means = state["means"] * jdecays + loss * (1.0 - jdecays)

        t = state["iteration"] + 1
        cor_mean = means / (1.0 - jdecays**t)

        running_min = jnp.minimum(state["running_min"], cor_mean)

        return {
            "means": means,
            "iteration": t,
            "running_min": running_min,
            "decays": jdecays,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(self, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Flatten loss statistics into a single feature vector:

          [debias_mean, debias_mean - running_min, running_min]  per timescale
        """
        jdecays = state["decays"]
        t = jnp.maximum(state["iteration"], 1)
        cor_mean = state["means"] / (1.0 - jdecays**t)
        running_min = state["running_min"]

        delta = cor_mean - running_min

        feat = jnp.stack([cor_mean, delta, running_min], axis=-1)  # [S, 3]
        return feat.reshape(-1)  # [S * 3]


# ----------------------------------------------------------------------
# Mnemo encoders: topological + temporal encoders
# ----------------------------------------------------------------------

import functools


class MnemoCore(hk.Module):
    """
    Core Mnemosyne-style encoder:

      * Input:
          feats: [L, D_feat]  (L = #parameter leaves, D_feat = per-leaf features)
          h_prev: [L, H]      (temporal state per leaf)
      * Output:
          lr_mult: [L]        (per-leaf scalar step-size multiplier)
          h_next: [L, H]      (updated temporal state)

    Architecture:
      1. Project feats -> hidden_dim.
      2. One layer of multi-head self-attention (topological encoder).
      3. One small residual MLP.
      4. Per-leaf temporal update: h_next = tanh(W_x x + W_h h_prev + b).
      5. Prediction head: lr_mult = exp( clamp(MLP(h_next)) ) * base_lr.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        base_lr: float = 1e-3,
        name: str = "mnemo_core",
    ):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.base_lr = base_lr

    def __call__(
        self, feats: jnp.ndarray, h_prev: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          feats: [L, D_feat]
          h_prev: [L, H]

        Returns:
          lr_mult: [L]
          h_next:  [L, H]
        """
        L, _ = feats.shape

        # 1) Project to shared hidden size.
        x = hk.Linear(self.hidden_dim, name="feat_proj")(feats)  # [L, H]

        # 2) Topological encoder via multi-head self-attention.
        x_attn_in = jnp.expand_dims(x, axis=1)  # [L, 1, H]
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.hidden_dim // max(self.num_heads, 1),
            name="topo_mha",
        )
        attn_out = attn(x_attn_in, x_attn_in, x_attn_in)  # [L, 1, H]
        attn_out = jnp.squeeze(attn_out, axis=1)  # [L, H]

        x = x + attn_out
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        # 3) Small residual MLP.
        mlp = hk.nets.MLP([self.hidden_dim, self.hidden_dim], name="topo_mlp")
        x_mlp = mlp(x)
        x = x + x_mlp
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        # 4) Temporal encoder: per-leaf RNN-style update.
        # h_next = tanh(W_x x + W_h h_prev + b)
        Wh = hk.get_parameter(
            "Wh",
            shape=(self.hidden_dim, self.hidden_dim),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        )
        Wx = hk.get_parameter(
            "Wx",
            shape=(self.hidden_dim, self.hidden_dim),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
        )
        b = hk.get_parameter(
            "b",
            shape=(self.hidden_dim,),
            dtype=x.dtype,
            init=jnp.zeros,
        )

        # x, h_prev: [L, H]
        h_next = jnp.tanh(x @ Wx + h_prev @ Wh + b)  # [L, H]

        # 5) Head: per-token log step-size, then exponentiate and scale.
        head = hk.nets.MLP([self.hidden_dim, 1], name="lr_head")
        lr_logits = head(h_next)  # [L, 1]
        lr_logits = jnp.squeeze(lr_logits, axis=-1)  # [L]

        # Clamp to prevent crazy step-sizes; then exponentiate.
        lr_logits = jnp.clip(lr_logits, -5.0, 5.0)
        lr_mult = jnp.exp(lr_logits) * self.base_lr  # [L]

        return lr_mult, h_next


# ----------------------------------------------------------------------
# Inner optimizer state
# ----------------------------------------------------------------------


@flax.struct.dataclass
class State:
    """Inner optimizer state for Mnemo-CeLo."""

    params: chex.ArrayTree
    state: chex.ArrayTree  # model (optimizee) state
    iteration: jnp.ndarray
    num_steps: jnp.ndarray
    temporal_state: chex.ArrayTree  # PyTree, same structure as params, leaf shape [H]
    loss_buffer: chex.ArrayTree


# ----------------------------------------------------------------------
# Utility: per-parameter feature construction
# ----------------------------------------------------------------------


def _per_param_features(p: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """
    Summarize a parameter tensor and its gradient into a small feature vector.

    Features (all scalars):
      - mean(g)
      - mean(|g|)
      - log( mean(g^2) + eps )
      - log( mean(|p|) + eps )
    """
    p = jnp.asarray(p)
    g = jnp.asarray(g)

    g_flat = jnp.reshape(g, [-1])
    p_flat = jnp.reshape(p, [-1])

    eps = 1e-8
    mean_g = jnp.mean(g_flat)
    mean_abs_g = jnp.mean(jnp.abs(g_flat))
    log_mean_sq_g = jnp.log(jnp.mean(g_flat**2) + eps)
    log_mean_abs_p = jnp.log(jnp.mean(jnp.abs(p_flat)) + eps)

    return jnp.stack([mean_g, mean_abs_g, log_mean_sq_g, log_mean_abs_p], axis=0)


# ----------------------------------------------------------------------
# Mnemo-CeLo learned optimizer
# ----------------------------------------------------------------------


@gin.configurable
class MnemoCelo(lopt_base.LearnedOptimizer):
    """
    Mnemosyne-style learned optimizer with CeLo-compatible interface.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_loss_scales: int = 8,
        base_lr: float = 1e-3,
        param_inits: int = 1,  
        validation_mode: bool = False,
        summarize_each_layer: bool = False,
        summarize_all_control: bool = False,
        train_phase: int = 2,
    ):
        """
        Args:
          hidden_dim: size of temporal/attn hidden state H.
          num_heads: number of attention heads in the topological encoder.
          num_loss_scales: number of timescales for the loss buffer.
          base_lr: base step size scale (the network predicts a multiplicative factor).
          param_inits: kept for compatibility with CeLo; not used here.
          validation_mode: if True, we expect `loss` to be passed into update,
                           but the optimizer behaviour is otherwise identical.
          summarize_each_layer: currently unused; kept for compatibility.
          summarize_all_control: currently unused; kept for compatibility.
          train_phase: 1 or 2, if you want to mimic CeLo's phase-1/phase-2
                      meta-training.
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_loss_scales = num_loss_scales
        self.base_lr = base_lr
        self.param_inits = param_inits
        self.validation_mode = validation_mode
        self.summarize_each_layer = summarize_each_layer
        self.summarize_all_control = summarize_all_control
        self.train_phase = train_phase

        # Loss buffer used for temporal progress / schedule features.
        self.buffer_loss_fns = BufferLossAccumulators(num_scales=num_loss_scales)

        # ---- Haiku transform for the Mnemo core ----
        def core_forward(feats, h_prev):
            core = MnemoCore(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                base_lr=self.base_lr,
            )
            return core(feats, h_prev)

        # We don't need an RNG at apply time; all randomness is in the
        # initialization of parameters.
        self.core = hk.without_apply_rng(hk.transform(core_forward))

    # ------------------------------------------------------------------
    # LearnedOptimizer API
    # ------------------------------------------------------------------

    def init(self, key) -> lopt_base.MetaParams:
        """
        Initialize meta-parameters (theta) of the learned optimizer.

        We use dummy inputs to instantiate the Haiku network parameters.
        """
        # Dummy shapes: a small number of "parameter tokens", each with the
        # fixed feature dimension we use at runtime.
        dummy_L = 4
        local_feat_dim = 4 
        loss_feat_dim = self.num_loss_scales * 3  
        extra_feat_dim = 1  
        feat_dim = local_feat_dim + loss_feat_dim + extra_feat_dim

        dummy_feats = jnp.zeros((dummy_L, feat_dim), dtype=jnp.float32)
        dummy_h = jnp.zeros((dummy_L, self.hidden_dim), dtype=jnp.float32)

        core_params = self.core.init(key, dummy_feats, dummy_h)

        theta = {"core_params": core_params}
        return theta

    def opt_fn(self, theta, is_training: bool = True) -> opt_base.Optimizer:
        """
        Returns an optax-style inner optimizer, matching CeLo's pattern:
          - .init(params, model_state=None, num_steps=None, key=None) -> State
          - .update(state, grad, loss=None, model_state=None, key=None) -> State
        """
        parent = self

        class _Opt(opt_base.Optimizer):
            """Inner Mnemo-CeLo optimizer."""

            def __init__(self, theta_inner):
                super().__init__()
                self.theta = theta_inner

            @functools.partial(jax.jit, static_argnums=(0,))
            def init(
                self,
                params: Any,
                model_state: Any = None,
                num_steps: int = None,
                key=None,
            ) -> State:
                if num_steps is None:
                    # Fall back to a large-ish horizon if not provided.
                    num_steps_val = 10000
                else:
                    num_steps_val = int(num_steps)

                # Temporal state: same tree structure as params, each leaf [H]
                def _init_h(_) -> jnp.ndarray:
                    return jnp.zeros((parent.hidden_dim,), dtype=jnp.float32)

                temporal_state = jax.tree_util.tree_map(_init_h, params)

                loss_buffer = parent.buffer_loss_fns.init(num_steps_val)

                return State(
                    params=params,
                    state=model_state,
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    num_steps=jnp.asarray(num_steps_val, dtype=jnp.int32),
                    temporal_state=temporal_state,
                    loss_buffer=loss_buffer,
                )

            @functools.partial(jax.jit, static_argnums=(0,))
            def update(
                self,
                opt_state: State,
                grad: Any,
                loss: jnp.ndarray = None,
                model_state: Any = None,
                key=None,
            ) -> State:
                # grad PyTree must match params structure.
                grads = grad

                # Update loss buffer and get global loss features.
                if loss is None:
                    # If no loss is passed (e.g., some meta-training setups),
                    # we just reuse the previous loss buffer.
                    loss_buffer = opt_state.loss_buffer
                else:
                    loss_buffer = parent.buffer_loss_fns.update(
                        opt_state.loss_buffer, loss
                    )
                loss_feats = parent.buffer_loss_fns.features(loss_buffer)  # [S*3]

                # Normalized training progress.
                frac_trained = opt_state.iteration / jnp.maximum(
                    opt_state.num_steps, 1
                )
                frac_trained = jnp.asarray(frac_trained, dtype=jnp.float32)

                # Flatten params, grads, temporal_state into aligned lists.
                flat_params, treedef = jax.tree_util.tree_flatten(opt_state.params)
                flat_grads = treedef.flatten_up_to(grads)
                flat_h = treedef.flatten_up_to(opt_state.temporal_state)

                # Build per-parameter local features.
                local_feats = [
                    _per_param_features(p, g) for p, g in zip(flat_params, flat_grads)
                ]  # list of [4]

                # Tile global features & concat.
                loss_feats_tiled = [
                    loss_feats for _ in local_feats
                ]  # each [loss_feat_dim]
                frac_feat = jnp.expand_dims(frac_trained, axis=0)  # [1]

                per_token_feats = [
                    jnp.concatenate([lf, lsf, frac_feat], axis=0)
                    for lf, lsf in zip(local_feats, loss_feats_tiled)
                ]  # each [feat_dim]

                feats_stack = jnp.stack(per_token_feats, axis=0)  # [L, feat_dim]
                h_prev_stack = jnp.stack(flat_h, axis=0)  # [L, H]

                # Run Mnemo core network.
                lr_mult, h_next_stack = parent.core.apply(
                    self.theta["core_params"], feats_stack, h_prev_stack
                )  # lr_mult: [L], h_next_stack: [L, H]

                # Map lr_mult and h_next back to PyTree structure.
                lr_list = list(lr_mult)
                h_list = [h_next_stack[i] for i in range(h_next_stack.shape[0])]

                lr_tree = treedef.unflatten(lr_list)
                h_tree = treedef.unflatten(h_list)

                # Simple per-parameter update: p_new = p - lr * g
                def _apply_update(p, g, lr_scalar):
                    g = jnp.asarray(g)
                    lr_scalar = jnp.asarray(lr_scalar, dtype=g.dtype)
                    # Optional gradient clipping for safety.
                    g = jnp.clip(g, -1000.0, 1000.0)
                    return p - lr_scalar * g

                next_params = jax.tree_util.tree_map(
                    _apply_update, opt_state.params, grads, lr_tree
                )

                next_state = State(
                    params=next_params,
                    state=model_state,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps,
                    temporal_state=h_tree,
                    loss_buffer=loss_buffer,
                )

                # Ensure we preserve PyTree structure / type.
                return tree_utils.match_type(next_state, opt_state)

        return _Opt(theta)
