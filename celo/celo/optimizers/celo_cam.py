# celo/optimizers/celo_cam.py
# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Celo-CAM learned optimizer (tensor-level temporal encoder).

This variant replaces the LSTM temporal module with a Compact Associative Memory (CAM)
as described in Mnemosyne Sec. 3.3 (Eqs. (2)-(5)), using a softmax-kernel linearization
via hyperbolic-cosine random features and exponential discounting.

Pipeline:
 - Build the same per-tensor feature vector as the original Celo (no change).
 - Feed the feature stream through CAM (updates hidden state (N_t, Psi_t) per step).
 - From the CAM-augmented features, predict control weights and lr multipliers.
 - Interpolate per-parameter MLP weights with those control weights (unchanged).
 - Produce parameter updates with the same MLP (`ff_mod`) as before (unchanged).
"""

import functools
from typing import Any, Optional, Sequence, Tuple

import chex
import flax
import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
from absl import logging
from jax import lax
from learned_optimization import summary, tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base

# ------------------------------
# Utility / unchanged helpers
# ------------------------------

def _fractional_tanh_embed(x):
    def one_freq(timescale):
        return jnp.tanh((x - (jnp.float32(timescale))) * 10)
    timescales = jnp.asarray([0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1], dtype=jnp.float32)
    return jax.vmap(one_freq)(timescales)

def factored_dims(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    if len(shape) < 2:
        return None
    sorted_dims = onp.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])

def _safe_rsqrt(x):
    return lax.rsqrt(jnp.maximum(x, 1e-9))

def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))

def _sorted_values(dd):
    return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]

def _clip_log_abs(v, scale=1.0):
    mag = jnp.log(1e-20 + jnp.abs(v))
    return jnp.clip(mag, -50.0, 50.0) * 0.5

class BufferLossAccumulators:
    """Rolling accumulator for loss values."""
    def __init__(self): pass

    def init(self, num_steps):
        halflife = jnp.logspace(1, jnp.log10(num_steps), 10)
        decays = jnp.exp(-1.0 / halflife)
        return {
            "means": jnp.zeros((len(decays),), dtype=jnp.float32),
            "iteration": jnp.asarray(0, dtype=jnp.int32),
            "running_min": 999999999999.0 * jnp.ones((len(decays),), dtype=jnp.float32),
            "decays": decays,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, state, loss):
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"] + 1))
        approx_max = jnp.max(cor_mean)
        approx_max = jnp.where(state["iteration"] == 0, loss, approx_max)
        loss = jnp.minimum(jnp.abs(approx_max) * 2, loss)
        means = state["means"] * jdecays + loss * (1.0 - jdecays)
        cor_mean = means / (1 - jdecays ** (state["iteration"] + 1))
        running_min = jnp.minimum(state["running_min"], cor_mean)
        return {
            "means": means,
            "iteration": state["iteration"] + 1,
            "running_min": running_min,
            "decays": state["decays"],
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(self, state):
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"]))
        approx_max = cor_mean[1:]
        cor_mean = cor_mean[0:-1]
        running_min = state["running_min"][0:-1]
        den = jnp.maximum(1e-8, (approx_max - running_min))
        pre_center = (cor_mean - running_min) / den
        feature1 = jnp.clip(pre_center - 1.0, -1, 1)
        return jnp.where(state["iteration"] <= 2, feature1 * 0, feature1)

@flax.struct.dataclass
class State:
    """Inner state of learned optimizer."""
    params: chex.ArrayTree
    rms_rolling: chex.ArrayTree
    mom_rolling: chex.ArrayTree
    fac_rolling: chex.ArrayTree
    iteration: jnp.ndarray
    state: chex.ArrayTree
    num_steps: jnp.ndarray
    loss_buffer: chex.ArrayTree
    # CAM hidden state (shared across tensors / time)
    cam_N: jnp.ndarray      # (r_feats, feat_dim)
    cam_Psi: jnp.ndarray    # (r_feats,)

# ------------------------------
# Celo-CAM Optimizer
# ------------------------------

@gin.configurable
class Celo(lopt_base.LearnedOptimizer):
    """
    Celo optimizer using CAM as the temporal encoder (tensor-wise).
    - Temporal encoder: CAM (Sec. 3.3), exponential discount, cosh RFs (Appendix A.1).
    - Spatial/per-parameter MLP: identical to Celo (ff_mod).
    """

    def __init__(
        self,
        # FF (unchanged)
        ff_hidden_size=4,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        param_inits=64,
        mix_layers=True,
        exp_mult=0.001,
        step_mult=0.001,
        validation_mode=False,
        with_validation_feature_dim=False,
        train_phase=1,
        # feature ablations (unchanged)
        with_g=True,
        with_m=True,
        with_m_feat=True,
        with_rms=True,
        with_rms_feat=True,
        with_rms_norm_g=True,
        with_rsqrt_rms=True,
        with_p=True,
        with_fac_norm_g=True,
        with_fac_rms=True,
        with_fac_rsqrt=True,
        with_grad_clip_feat=True,
        with_fac_mom_mult=True,
        with_rms_only_norm_g=True,
        adafactor_accumulator=True,
        param_scale_mult=True,
        precondition_output=False,
        reparam_decay=10.0,
        # more summaries
        summarize_each_layer=False,
        summarize_all_control=False,
        # Debug/behavior probes
        constant_loss=False,
        clip_param_scale_amount=None,

        # -------------------------
        # CAM hyperparameters
        # -------------------------
        cam_n_proj=64,        # N: q/k projection size
        cam_r_feats=128,      # r: number of RFs (must be even)
        cam_tau=0.02,         # exponential discount τ
    ):
        super().__init__()
        # Save core settings
        self.ff_hidden_size = ff_hidden_size
        self.ff_hidden_layers = ff_hidden_layers
        self.initial_momentum_decays = initial_momentum_decays
        self.initial_rms_decays = initial_rms_decays
        self.initial_adafactor_decays = initial_adafactor_decays
        self.param_inits = param_inits
        self.mix_layers = mix_layers
        self.with_g = with_g
        self.with_m = with_m
        self.with_m_feat = with_m_feat
        self.with_rms = with_rms
        self.with_rms_feat = with_rms_feat
        self.with_rms_norm_g = with_rms_norm_g
        self.with_rsqrt_rms = with_rsqrt_rms
        self.with_p = with_p
        self.with_fac_norm_g = with_fac_norm_g
        self.with_fac_rms = with_fac_rms
        self.with_fac_rsqrt = with_fac_rsqrt
        self.with_grad_clip_feat = with_grad_clip_feat
        self.with_fac_mom_mult = with_fac_mom_mult
        self.with_rms_only_norm_g = with_rms_only_norm_g
        self.adafactor_accumulator = adafactor_accumulator
        self.param_scale_mult = param_scale_mult
        self.exp_mult = exp_mult
        self.step_mult = step_mult
        self.summarize_each_layer = summarize_each_layer
        self.precondition_output = precondition_output
        self.reparam_decay = reparam_decay
        self.with_validation_feature_dim = with_validation_feature_dim
        self.validation_mode = validation_mode
        self.constant_loss = constant_loss
        self.summarize_all_control = summarize_all_control
        self.clip_param_scale_amount = clip_param_scale_amount
        self.train_phase = train_phase

        # CAM hparams
        assert cam_r_feats % 2 == 0, "cam_r_feats must be even for cosh RFs."
        self.cam_n_proj = cam_n_proj
        self.cam_r_feats = cam_r_feats
        self.cam_tau = cam_tau
        self.cam_decay = jnp.exp(-jnp.asarray(cam_tau))

        logging.info(
            f"[Celo-CAM] Validation mode: {self.validation_mode} "
            f"(with valid feature dim: {with_validation_feature_dim})"
        )

        # Per-parameter MLP and small helpers: unchanged
        self.ff_mod = hk.transform(self._ff_mod)
        self.buffer_loss_fns = BufferLossAccumulators()

    # --------------------------
    # Feature construction: unchanged
    # --------------------------
    def _decay_to_param(self, x):
        return jnp.log(1 - x) / self.reparam_decay

    def _param_to_decay(self, x):
        return 1 - jnp.exp(x * self.reparam_decay)

    def accumulators_for_decays(self, mom_param=None, rms_param=None, adafactor_param=None):
        if mom_param is None:
            mom_decay = jnp.asarray(self.initial_momentum_decays)
        else:
            mom_decay = self._param_to_decay(
                self._decay_to_param(jnp.asarray(self.initial_momentum_decays)) + mom_param
            )
        if rms_param is None:
            rms_decay = jnp.asarray(self.initial_rms_decays)
        else:
            rms_decay = self._param_to_decay(
                self._decay_to_param(jnp.asarray(self.initial_rms_decays)) + rms_param
            )
        if adafactor_param is None:
            adafactor_decay = jnp.asarray(self.initial_adafactor_decays)
        else:
            adafactor_decay = self._param_to_decay(
                self._decay_to_param(jnp.asarray(self.initial_adafactor_decays)) + adafactor_param
            )

        mom_roll = common.vec_rolling_mom(mom_decay)
        rms_roll = common.vec_rolling_rms(rms_decay)
        fac_vec_roll = common.vec_factored_rolling(adafactor_decay)
        return mom_roll, rms_roll, fac_vec_roll

    def lstm_features_for_tensor(  # name kept for compatibility
        self, p, g, m, rms, summary_prefix, fraction_trained, loss_features
    ):
        inputs = {}
        fraction_left = _fractional_tanh_embed(fraction_trained)
        inputs["fraction_left"] = fraction_left
        inputs["loss_features"] = loss_features

        if self.summarize_each_layer:
            for k, v in inputs.items():
                if len(v.shape) > 0:
                    for vi, vv in enumerate(v):
                        summary.summary(f"per_tensor_feat/{summary_prefix}/{k}__{vi}", vv, aggregation="sample")
                else:
                    summary.summary(f"per_tensor_feat/{summary_prefix}/{k}", v, aggregation="sample")

        values = _sorted_values(inputs)
        values = [v if len(v.shape) == 1 else jnp.expand_dims(v, 0) for v in values]

        if self.with_validation_feature_dim:
            values.append(jnp.ones([1], dtype=jnp.float32) * self.validation_mode)

        return jnp.concatenate(values, axis=0)

    # --------------------------
    # Per-parameter MLP: unchanged
    # --------------------------
    def _ff_mod(
        self,
        global_feat,
        extra_step_mult,
        p,
        g,
        m,
        rms,
        fac_g,
        fac_vec_col,
        fac_vec_row,
        fac_vec_v,
        summary_prefix,
    ):
        if len(p.shape) == 0:
            p = jnp.expand_dims(p, 0); g = jnp.expand_dims(g, 0); m = jnp.expand_dims(m, 0)
            rms = jnp.expand_dims(rms, 0); fac_g = jnp.expand_dims(fac_g, 0)
            fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
            fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
            fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
            did_reshape = True
        else:
            did_reshape = False

        inps = []
        if self.with_g: inps.append(jnp.expand_dims(g, axis=-1))
        if self.with_grad_clip_feat: inps.append(jnp.expand_dims(jnp.clip(g, -0.1, 0.1), axis=-1))
        if self.with_p: inps.append(jnp.expand_dims(p, axis=-1))
        if self.with_m and self.with_m_feat: inps.append(m)
        if self.with_rms and self.with_rms_feat: inps.append(rms)

        if self.with_rms_norm_g or self.with_rsqrt_rms and self.with_rms_only_norm_g:
            rsqrt = lax.rsqrt(rms + 1e-6)
        if self.with_rms_norm_g: inps.append(m * rsqrt)
        if self.with_rsqrt_rms: inps.append(rsqrt)
        if self.with_fac_norm_g: inps.append(fac_g)
        if self.with_rms_only_norm_g:
            rms_norm_g = jnp.expand_dims(g, axis=-1) * rsqrt
            inps.append(rms_norm_g)

        if self.adafactor_accumulator:
            factored_dim = factored_dims(g.shape)
            if factored_dim is not None:
                d1, d0 = factored_dim
                to_tile = [1] * (1 + len(g.shape)); to_tile[d0] = g.shape[d0]
                row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)
                to_tile = [1] * (1 + len(g.shape)); to_tile[d1] = g.shape[d1]
                col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

                if self.with_fac_rms: inps += [row_feat, col_feat]
                if self.with_fac_rsqrt:
                    inps += [lax.rsqrt(row_feat + 1e-8), lax.rsqrt(col_feat + 1e-8)]

                reduced_d1 = d1 - 1 if d1 > d0 else d1
                row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)
                row_factor = _safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
                col_factor = _safe_rsqrt(fac_vec_col)
                if self.with_fac_mom_mult:
                    fac_mom_mult = m * jnp.expand_dims(row_factor, axis=d0) * jnp.expand_dims(col_factor, axis=d1)
                    inps.append(fac_mom_mult)
            else:
                if self.with_fac_rms: inps += [fac_vec_v, fac_vec_v]
                if self.with_fac_rsqrt: inps += [lax.rsqrt(fac_vec_v + 1e-8), lax.rsqrt(fac_vec_v + 1e-8)]
                if self.with_fac_mom_mult: inps.append(m * (fac_vec_v) ** -0.5)

        last_size = sum([i.shape[-1] for i in inps])
        weights, biases = [], []
        for wi, w in enumerate([self.ff_hidden_size] * (self.ff_hidden_layers) + [3]):
            stddev = 1.0 / onp.sqrt(last_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            if wi == 0:
                w1 = []
                for ii, i in enumerate(inps):
                    w1.append(hk.get_parameter(f"w{wi}__{ii}", shape=(i.shape[-1], w), dtype=jnp.float32, init=w_init))
                weights.append(w1)
            else:
                weights.append(hk.get_parameter(f"w{wi}", shape=(last_size, w), dtype=jnp.float32, init=w_init))
            biases.append(hk.get_parameter(f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros))
            last_size = w

        axis = list(range(len(p.shape)))
        inp_stack = [_second_moment_normalizer(i, axis=axis) for i in inps]

        o = inp_stack
        for wi, (w, b) in enumerate(zip(weights, biases)):
            if wi == 0:
                o_tmp = jnp.zeros(o[0].shape[:-1] + w[0].shape[1:])
                for oi, oo in enumerate(o):
                    o_tmp = o_tmp + oo @ w[oi]
            else:
                o_tmp = o @ w
            o = o_tmp + jnp.broadcast_to(b, list(o_tmp.shape[0:-1]) + [o_tmp.shape[-1]])
            if wi != len(weights) - 1:
                o = jax.nn.relu(o)

        direction = o[..., 0]
        magnitude_param = o[..., 1]
        mag_param = jnp.exp(magnitude_param * self.exp_mult)
        param_scale = jnp.sqrt(jnp.mean(jnp.square(p)) + 1e-9)
        summary.summary(f"celo/{summary_prefix}/param_scale", param_scale)

        if self.clip_param_scale_amount is not None:
            max_scale = self.clip_param_scale_amount * onp.sqrt(onp.prod(p.shape))
            param_scale = jnp.minimum(param_scale, jnp.asarray(max_scale, dtype=jnp.float32))
            summary.summary(f"celo/{summary_prefix}/post_param_scale", param_scale)

        avg_step_size = jnp.mean(jnp.abs(direction * mag_param * self.step_mult * extra_step_mult))
        summary.summary(f"celo/{summary_prefix}/no_parammag_mult_avg_step_size", avg_step_size)

        if self.param_scale_mult:
            step = direction * (param_scale * mag_param) * self.step_mult
        else:
            step = direction * mag_param * self.step_mult

        if self.train_phase > 1:
            step = extra_step_mult * step

        avg_step_size = jnp.mean(jnp.abs(step))
        summary.summary(f"celo/{summary_prefix}/pre_precondition_avg_step_size", avg_step_size)

        step = step.reshape(p.shape)
        if self.precondition_output:
            norms = jax.tree_util.tree_map(lambda x: x[..., -1], rms)
            assert norms.shape == step.shape
            step = step * lax.rsqrt(norms + 1e-6)

        avg_step_size = jnp.mean(jnp.abs(step))
        summary.summary(f"celo/{summary_prefix}/avg_step_size", avg_step_size)
        summary.summary(f"celo/{summary_prefix}/extra_step_mult", extra_step_mult)

        new_p = p - step
        if did_reshape: new_p = jnp.squeeze(new_p, 0)
        return new_p

    # --------------------------
    # CAM core (JAX, stateless params; state carried in optimizer State)
    # --------------------------
    @staticmethod
    def _phi_cosh(z, omega):
        """
        Hyperbolic-cosine random features (Appendix A.1).
        z: (B, N)
        omega: (r/2, N)
        returns φ(z): (B, r)
        """
        half = omega.shape[0]
        proj = jnp.dot(z, omega.T)  # (B, half)
        m = jnp.max(jnp.abs(proj), axis=-1, keepdims=True)
        exp_pos = jnp.exp(proj - m)
        exp_neg = jnp.exp(-proj - m)
        r = 2 * half
        gamma = (1.0 / jnp.sqrt(r)) * jnp.exp(-0.5 * jnp.sum(z * z, axis=-1, keepdims=True) + m.squeeze(-1))
        phi = jnp.concatenate([exp_pos, exp_neg], axis=-1) * gamma
        return phi

    @staticmethod
    def _cam_update_read(x, cam_params, cam_state, decay):
        """
        One CAM step on a batch x (each row is a tensor-level feature vector).
        - x: (B, feat)
        - cam_params: dict with WQ, WK, WV, omega, head_control, head_lr
        - cam_state: (N_t, Psi_t) with N_t:(r, feat), Psi_t:(r,)
        Returns:
          x_prime: (B, feat), new_state:(N_t, Psi_t)
        """
        WQ, WK, WV = cam_params["WQ"], cam_params["WK"], cam_params["WV"]
        omega = cam_params["omega"]          # (r/2, N)
        N_t, Psi_t = cam_state

        q = jnp.dot(x, WQ)                  # (B, N)
        k = jnp.dot(x, WK)                  # (B, N)
        v = jnp.dot(x, WV)                  # (B, feat) -> v has same dim as x

        phi_k = Celo._phi_cosh(k, omega)    # (B, r)

        # Sequential stepwise decay per Eq. (4); loop over B to match exp(-tau) per pattern
        def body_fun(i, carry):
            N_cur, Psi_cur = carry
            pk = phi_k[i]                   # (r,)
            vi = v[i]                       # (feat,)
            N_next = decay * N_cur + jnp.outer(pk, vi)
            Psi_next = decay * Psi_cur + pk
            return (N_next, Psi_next)

        (N_new, Psi_new) = lax.fori_loop(0, x.shape[0], body_fun, (N_t, Psi_t))

        # Read Eq. (5)
        phi_q = Celo._phi_cosh(q, omega)    # (B, r)
        num = jnp.dot(phi_q, N_new)         # (B, feat)
        den = jnp.dot(phi_q, Psi_new)       # (B,)
        den = jnp.maximum(den, 1e-12)[:, None]
        delta = num / den                   # (B, feat)
        x_prime = x + delta
        return x_prime, (N_new, Psi_new)

    # --------------------------
    # Init meta-parameters (theta) and optimizer state
    # --------------------------
    def init(self, key) -> lopt_base.MetaParams:
        # Build ff_mod weights (unchanged)
        r = 10; c = 10
        p = jnp.ones([r, c]); g = jnp.ones([r, c])
        m = jnp.ones([r, c, len(self.initial_momentum_decays)])
        rms = jnp.ones([r, c, len(self.initial_rms_decays)])
        fac_g = jnp.ones([r, c, len(self.initial_adafactor_decays)])
        fac_vec_row = jnp.ones([r, len(self.initial_adafactor_decays)])
        fac_vec_col = jnp.ones([c, len(self.initial_adafactor_decays)])
        fac_vec_v = jnp.ones([len(self.initial_adafactor_decays)])

        def ffmod_init(key):
            global_features = {"iterations": 0, "num_steps": 10}
            mod_theta = self.ff_mod.init(
                key, global_features, 1.0, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row, fac_vec_v, 0.0
            )
            return mod_theta

        key1, key = jax.random.split(key)
        per_param_thetas = jax.vmap(ffmod_init)(jax.random.split(key1, self.param_inits))

        # Determine tensor-level feature dimension (unchanged helper)
        loss_features = self.buffer_loss_fns.features(self.buffer_loss_fns.init(10))
        output_shape = jax.eval_shape(
            self.lstm_features_for_tensor,  # same function name, but not using LSTM
            p, p, m, rms, 0,  # summary_prefix dummy
            fraction_trained=1.0,
            loss_features=loss_features,
        )
        feat_dim = int(output_shape.shape[0])

        # Initialize CAM meta-parameters
        N = self.cam_n_proj
        r_feats = self.cam_r_feats
        half = r_feats // 2

        def randn(k, shape, scale=0.01):
            return jax.random.normal(k, shape) * scale

        key, kWQ, kWK, kWV, kOMG, kHC, kHL = jax.random.split(key, 7)
        cam_params = {
            "WQ": randn(kWQ, (feat_dim, N), scale=onp.sqrt(1.0 / max(1, feat_dim))),
            "WK": randn(kWK, (feat_dim, N), scale=onp.sqrt(1.0 / max(1, feat_dim))),
            "WV": randn(kWV, (feat_dim, feat_dim), scale=onp.sqrt(1.0 / max(1, feat_dim))),
            "omega": randn(kOMG, (half, N), scale=1.0),  # RF directions ~ N(0, I)
            # Heads mapping CAM-enhanced features -> controls & lr_mult
            "head_control": randn(kHC, (feat_dim, self.param_inits), scale=onp.sqrt(1.0 / max(1, feat_dim))),
            "head_lr": randn(kHL, (feat_dim, 1), scale=onp.sqrt(1.0 / max(1, feat_dim))),
        }

        return {
            "ff_mod_stack": per_param_thetas,
            "cam_params": cam_params,
            "feat_dim": jnp.asarray(feat_dim, dtype=jnp.int32),
        }

    def get_frozen_param_keys(self):
        # Keep optional training phases consistent with original interface.
        if self.train_phase and self.train_phase == 1:
            return ["cam_params", "feat_dim"]  # freeze CAM, train ff_mod
        elif self.train_phase and self.train_phase == 2:
            return ["ff_mod_stack"]            # freeze ff_mod, train CAM
        else:
            return None

    def opt_fn(self, theta, is_training=True) -> opt_base.Optimizer:
        parent = self

        class _Opt(opt_base.Optimizer):
            """Inner optimizer using CAM temporal encoder."""

            def __init__(self, theta):
                super().__init__()
                self.theta = theta

            @functools.partial(jax.jit, static_argnums=(0,))
            def init(self, params: Any, model_state=None, num_steps=None, key=None) -> State:
                mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
                loss_buffer = parent.buffer_loss_fns.init(num_steps)

                feat_dim = int(self.theta["feat_dim"])
                r_feats = parent.cam_r_feats
                cam_N0 = jnp.zeros((r_feats, feat_dim), dtype=jnp.float32)
                cam_Psi0 = jnp.zeros((r_feats,), dtype=jnp.float32)

                return State(
                    params=params,
                    state=model_state,
                    rms_rolling=rms_roll.init(params),
                    mom_rolling=mom_roll.init(params),
                    fac_rolling=adafac_roll.init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
                    loss_buffer=loss_buffer,
                    cam_N=cam_N0,
                    cam_Psi=cam_Psi0,
                )

            @functools.partial(jax.jit, static_argnums=(0,))
            def update(
                self,
                opt_state,
                grads,
                loss=None,
                model_state=None,
                is_valid=False,
                key=None,
            ) -> State:
                if parent.constant_loss:
                    loss = 1.0
                assert loss is not None
                summary.summary("validation_mode", parent.validation_mode)

                # Update loss features
                next_loss_buffer = parent.buffer_loss_fns.update(opt_state.loss_buffer, loss)
                to_cam_from_loss = parent.buffer_loss_fns.features(next_loss_buffer)

                grads = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1000.0, 1000.0), grads)

                # Build tensor-level features (same as original "lstm_features_for_tensor")
                fraction_trained = opt_state.iteration / jnp.asarray(opt_state.num_steps, dtype=jnp.float32)
                ff = functools.partial(
                    parent.lstm_features_for_tensor,
                    fraction_trained=fraction_trained,
                    loss_features=to_cam_from_loss,
                )

                m = opt_state.mom_rolling.m
                rms = opt_state.rms_rolling.rms
                if parent.summarize_each_layer:
                    summary_prefix = tree_utils.map_named(lambda k, v: k, opt_state.params)
                else:
                    summary_prefix = jax.tree_util.tree_map(lambda x: "None", opt_state.params)

                rnn_inputs = jax.tree_util.tree_map(ff, opt_state.params, grads, m, rms, summary_prefix)
                stack = jnp.asarray(jax.tree_util.tree_leaves(rnn_inputs))  # (num_tensors, feat_dim)

                # ---- CAM temporal encoder step (Eq. (4), (5)) ----
                cam_params = self.theta["cam_params"]
                x_prime, (N_new, Psi_new) = parent._cam_update_read(
                    stack, cam_params, (opt_state.cam_N, opt_state.cam_Psi), parent.cam_decay
                )

                # Heads: controls + lr_mult from CAM-enhanced features
                logits = jnp.dot(x_prime, cam_params["head_control"])     # (num_tensors, param_inits)
                control_params = jax.nn.softmax(logits, axis=-1)          # list per tensor later

                lr_raw = jnp.dot(x_prime, cam_params["head_lr"]).squeeze(-1)  # (num_tensors,)
                lr_mult = jnp.exp(lr_raw) * 0.1

                if parent.summarize_all_control:
                    for pi in range(control_params.shape[0]):
                        summary.summary(f"control_param/{pi}", control_params[pi], "tensor")

                # Per-tensor accumulators update (unchanged)
                mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
                next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grads)
                next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grads)
                next_adafac_rolling, fac_g = adafac_roll.update(opt_state.fac_rolling, grads)

                global_features = {"iterations": opt_state.iteration, "num_steps": opt_state.num_steps}

                # Interpolate per-parameter MLP weights with control weights (unchanged)
                def apply_one(
                    control_weight_vec,  # (param_inits,)
                    lr_m,
                    p,
                    g,
                    m,
                    rms,
                    fac_g,
                    v_col,
                    v_row,
                    v,
                    summary_prefix,
                ):
                    # interpolate theta["ff_mod_stack"] with softmax weights
                    def interpolate_theta(ff_p):
                        target = [ff_p.shape[0]] + [1] * (len(ff_p.shape) - 1)
                        c = jnp.reshape(control_weight_vec, target)
                        return jnp.mean(ff_p * c, axis=0)

                    ff_param = jax.tree_util.tree_map(interpolate_theta, self.theta["ff_mod_stack"])
                    next_p = parent.ff_mod.apply(
                        ff_param,
                        None,  # key not used by hk.transform without rng
                        global_features,
                        lr_m,
                        p,
                        g,
                        m=m,
                        rms=rms,
                        fac_g=fac_g,
                        fac_vec_col=v_col,
                        fac_vec_row=v_row,
                        fac_vec_v=v,
                        summary_prefix=summary_prefix,
                    )
                    return next_p

                # Unflatten control/lr to pytree structure of params
                struct = jax.tree_util.tree_structure(grads)
                ctl_list = list(jnp.split(control_params, control_params.shape[0], axis=0))
                ctl_list = [jnp.squeeze(c, axis=0) for c in ctl_list]  # each (param_inits,)
                lr_list = list(jnp.split(lr_mult, lr_mult.shape[0], axis=0))
                lr_list = [jnp.squeeze(c, axis=0) for c in lr_list]
                control_params_tree = struct.unflatten(ctl_list)
                lr_mult_tree = struct.unflatten(lr_list)

                next_params = jax.tree_util.tree_map(
                    apply_one,
                    control_params_tree,
                    lr_mult_tree,
                    opt_state.params,
                    grads,
                    next_mom_rolling.m,
                    next_rms_rolling.rms,
                    fac_g,
                    next_adafac_rolling.v_col,
                    next_adafac_rolling.v_row,
                    next_adafac_rolling.v_diag,
                    summary_prefix,
                )

                ss = State(
                    params=next_params,
                    state=model_state,
                    mom_rolling=next_mom_rolling,
                    rms_rolling=next_rms_rolling,
                    fac_rolling=next_adafac_rolling,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps,
                    loss_buffer=next_loss_buffer,
                    cam_N=N_new,
                    cam_Psi=Psi_new,
                )
                return tree_utils.match_type(ss, opt_state)

        return _Opt(theta)
