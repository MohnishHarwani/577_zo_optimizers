
# celo_compress.py
from __future__ import annotations
from typing import Any, Tuple
import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from celo.factory import get_optimizer
from celo.utils import load_state  # uses flax.serialization.from_bytes under the hood

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from typing import Any
from celo.utils import init_lopt_from_ckpt

# ------------------ Pruning ------------------
def _percentile_threshold(x: jnp.ndarray, keep_ratio: float) -> jnp.ndarray:
    """
    Compute the magnitude threshold so that ~keep_ratio of entries are kept.
    This is done with NumPy (on CPU) to avoid JAX static-arg hashing issues.
    """
    flat = np.abs(np.asarray(x)).ravel()
    N = flat.size
    if N == 0:
        # arbitrary, won't prune anything anyway
        return jnp.asarray(0.0, dtype=jnp.result_type(x))

    # number of elements to keep per array
    k = max(1, int(np.floor(keep_ratio * N)))
    # index (0-based) of the k-th largest value = N - k
    idx = max(0, min(N - 1, N - k))
    thresh_np = np.partition(flat, idx)[idx]
    return jnp.asarray(thresh_np, dtype=jnp.result_type(x))


def prune_by_magnitude(theta: Any, sparsity: float) -> Any:
    """Per-leaf magnitude pruning. sparsity=0.5 keeps ~50% largest per leaf."""
    # Optional: truly zero everything if sparsity >= 1.0
    if sparsity >= 1.0:
        def _all_zero(x):
            if not hasattr(x, "shape"):
                return x
            arr = jnp.asarray(x)
            if arr.size == 0 or arr.ndim == 0:
                return x
            return jnp.zeros_like(arr)
        return jtu.tree_map(_all_zero, theta)

    keep_ratio = 1.0 - sparsity

    def _prune(x):
        # Treat anything with a shape as array-like
        if not hasattr(x, "shape"):
            return x
        arr = jnp.asarray(x)
        # Skip scalars / empty arrays
        if arr.size == 0 or arr.ndim == 0:
            return x
        th = _percentile_threshold(arr, keep_ratio)
        pruned = jnp.where(jnp.abs(arr) >= th, arr, jnp.zeros_like(arr))
        return pruned

    return jtu.tree_map(_prune, theta)


# ------------------ Fake INT8 (per-array symmetric) ------------------
def quantize_fake_int8(theta: Any) -> Any:
    """Quantize each leaf to int8 with a per-array scale, then dequantize."""
    eps = 1e-8

    def _q(x):
        if not hasattr(x, "shape"):
            return x
        arr = jnp.asarray(x)
        if arr.size == 0 or arr.ndim == 0:
            return x
        s = jnp.max(jnp.abs(arr)) / 127.0 + eps
        q = jnp.round(arr / s).clip(-127, 127).astype(jnp.int8)
        return (q.astype(arr.dtype) * s).astype(arr.dtype)

    return jtu.tree_map(_q, theta)


# ------------------ bf16 cast ------------------
def cast_bf16(theta: Any) -> Any:
    def _c(x):
        if not hasattr(x, "shape"):
            return x
        arr = jnp.asarray(x)
        if arr.size == 0 or arr.ndim == 0:
            return x
        return arr.astype(jnp.bfloat16) if arr.dtype == jnp.float32 else arr

    return jtu.tree_map(_c, theta)


# --- paste into compress_celo.py ---

from typing import Any, Tuple, Literal, Optional
import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from celo.factory import get_optimizer
from celo.utils import load_state

# (keep your existing helpers: _percentile_threshold, prune_by_magnitude, quantize_fake_int8, cast_bf16)

Variant = Literal["baseline", "pruned", "quant8", "bf16"]

def _apply_variant(theta: Any, variant: Variant, sparsity: float = 0.5) -> Any:
    if variant == "baseline":
        return theta
    if variant == "pruned":
        return prune_by_magnitude(theta, sparsity=sparsity)
    if variant == "quant8":
        return quantize_fake_int8(theta)
    if variant == "bf16":
        return cast_bf16(theta)
    raise ValueError(f"Unknown variant: {variant}")

def build_celo_from_ckpt(ckpt_path: str,
                         variant: Variant = "baseline",
                         sparsity: float = 0.5) -> Tuple[Any, Any]:
    """Single-ckpt build (kept as-is for backward compatibility)."""
    lopt = get_optimizer("celo")  # phase-2 optimizer
    theta_template = lopt.init(jax.random.PRNGKey(0))
    theta = load_state(ckpt_path, theta_template)
    theta_c = _apply_variant(theta, variant, sparsity)
    opt = lopt.opt_fn(theta_c)
    return lopt, opt

def build_celo_two_stage(phase1_ckpt: str,
                         phase2_ckpt: str,
                         variant: Variant = "baseline",
                         sparsity: float = 0.5) -> Tuple[Any, Any]:

    """Two-stage init using phase1 + phase2 checkpoints.

    Notes:
    - At runtime, the optimizer used is phase-2 (`get_optimizer("celo")`).
    - We load phase1 weights for symmetry/checking (and future extensions),
      but the actual opt is constructed from the phase-2 theta.
    - Compression (prune/quant/bf16) is applied to the phase-2 theta.
    """
    # Load phase-1 (not strictly required to build the runtime opt, but allowed)
    lopt_p1 = get_optimizer("celo_phase1")
    theta_p1_tmpl = lopt_p1.init(jax.random.PRNGKey(0))
    _ = load_state(phase1_ckpt, theta_p1_tmpl)  # Loaded for validation/optionally inspect

    # Load phase-2 and build the actual optimizer
    lopt_p2 = get_optimizer("celo")
    theta_p2_tmpl = lopt_p2.init(jax.random.PRNGKey(1))
    theta_p2 = load_state(phase2_ckpt, theta_p2_tmpl)
    theta_p2_c = _apply_variant(theta_p2, variant, sparsity)
    opt = lopt_p2.opt_fn(theta_p2_c)
    return lopt_p2, opt

