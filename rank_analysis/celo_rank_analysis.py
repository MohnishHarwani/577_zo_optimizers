#!/usr/bin/env python3
"""
CeLO theta rank analysis via SVD.

Loads a theta.state checkpoint for the CeLO learned optimizer and computes
effective rank statistics for each matrix-shaped parameter leaf.

Usage:
    python celo_rank_analysis.py --ckpt ./theta.state --energy_tol 0.99
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from celo.factory import get_optimizer
from celo.utils import load_state


@dataclass
class RankCfg:
    ckpt_path: str = "./theta.state"
    energy_tol: float = 0.99  # fraction of spectral energy to keep for effective rank
    min_matrix_size: int = 16  # skip very tiny matrices


def _load_theta(ckpt_path: str) -> Any:
    """Load CeLO theta.state using the same pattern as your other scripts."""
    lopt = get_optimizer("celo")  # phase-2 CeLO optimizer
    theta_template = lopt.init(jax.random.PRNGKey(0))
    theta = load_state(ckpt_path, theta_template)
    return theta


def _iter_matrix_leaves(theta: Any, min_matrix_size: int):
    """
    Yield (index, original_shape, matrix_2d) for each leaf that can be treated
    as a matrix for SVD.
    """
    leaves, _ = jtu.tree_flatten(theta)

    for idx, leaf in enumerate(leaves):
        if not hasattr(leaf, "shape"):
            continue

        arr = jnp.asarray(leaf)
        if arr.size == 0:
            continue

        # We only consider things at least 2D as "matrices"
        if arr.ndim < 2:
            continue

        # Flatten all but the first dimension into columns
        if arr.ndim > 2:
            mat = arr.reshape(arr.shape[0], -1)
        else:
            mat = arr

        # Optionally enforce a minimum size (avoid tons of tiny SVDs)
        if mat.size < min_matrix_size:
            continue

        # For SVD we don't really care if it's (m, n) or (n, m), but
        # it's often nicer numerically to have m >= n.
        if mat.shape[0] < mat.shape[1]:
            mat = mat.T

        yield idx, arr.shape, mat


def _effective_rank_from_singular_values(
    s: jnp.ndarray, energy_tol: float
) -> Tuple[int, float]:
    """
    Given singular values s (nonnegative), compute:

    - energy_rank: smallest k such that sum_{i<=k} s_i^2 >= energy_tol * sum s_i^2
    - stable_rank: ||W||_F^2 / ||W||_2^2 = sum s_i^2 / max(s_i)^2
    """
    if s.size == 0:
        return 0, 0.0

    # Sort descending just to be safe (JAX's SVD already returns sorted)
    s = jnp.sort(s)[::-1]

    energy = s ** 2
    total_energy = energy.sum()
    if total_energy == 0:
        return 0, 0.0

    cum_energy = jnp.cumsum(energy)
    # index of first k where energy fraction >= energy_tol
    frac = cum_energy / total_energy
    k = int(jnp.searchsorted(frac, energy_tol)) + 1
    k = int(jnp.clip(k, 1, s.size))

    # Stable rank
    max_sv = s[0]
    stable_rank = float(total_energy / (max_sv ** 2))

    return k, stable_rank


def estimate_theta_ranks(cfg: RankCfg) -> Dict[str, Any]:
    """
    Load theta and compute SVD-based effective ranks for all matrix leaves.

    Returns a dict with per-leaf stats and global summaries.
    """
    theta = _load_theta(cfg.ckpt_path)

    total_params = 0
    for leaf in jtu.tree_leaves(theta):
        if hasattr(leaf, "size"):
            total_params += int(leaf.size)

    print(f"[INFO] Total optimizer parameters in theta: {total_params:,}")

    leaf_stats: List[Dict[str, Any]] = []

    for idx, orig_shape, mat in _iter_matrix_leaves(theta, cfg.min_matrix_size):
        m, n = mat.shape
        dof = min(m, n)

        # SVD on this matrix
        s = jnp.linalg.svd(mat, compute_uv=False)
        eff_rank, stable_rank = _effective_rank_from_singular_values(
            s, cfg.energy_tol
        )

        leaf_stats.append(
            dict(
                leaf_index=idx,
                orig_shape=tuple(int(x) for x in orig_shape),
                m=int(m),
                n=int(n),
                dof=int(dof),
                eff_rank=int(eff_rank),
                stable_rank=float(stable_rank),
                eff_rank_ratio=float(eff_rank / dof),
                stable_rank_ratio=float(stable_rank / dof),
            )
        )

    if not leaf_stats:
        print("[WARN] No matrix-shaped leaves found for rank analysis.")
        return {"total_params": total_params, "leaf_stats": []}

    # Print per-leaf summary
    print("\n[INFO] Per-matrix effective ranks (energy-based & stable rank):")
    for s in leaf_stats:
        print(
            f"  Leaf {s['leaf_index']:3d} "
            f"shape={s['orig_shape']}  "
            f"dof={s['dof']:4d}  "
            f"eff_rank≈{s['eff_rank']:4d} "
            f"({s['eff_rank_ratio']*100:5.1f}% of dof)  "
            f"stable_rank≈{s['stable_rank']:.2f} "
            f"({s['stable_rank_ratio']*100:5.1f}% of dof)"
        )

    # Global summaries
    avg_eff_ratio = sum(s["eff_rank_ratio"] for s in leaf_stats) / len(leaf_stats)
    avg_stable_ratio = sum(s["stable_rank_ratio"] for s in leaf_stats) / len(
        leaf_stats
    )

    print("\n[INFO] Global summary over matrix leaves:")
    print(
        f"  Mean (eff_rank / dof):   {avg_eff_ratio:.3f} "
        f"({avg_eff_ratio*100:.1f}% of full rank on average)"
    )
    print(
        f"  Mean (stable_rank / dof): {avg_stable_ratio:.3f} "
        f"({avg_stable_ratio*100:.1f}% of full rank on average)"
    )
    print(f"  Energy tolerance used for eff_rank: {cfg.energy_tol:.3f}")

    return {
        "total_params": total_params,
        "leaf_stats": leaf_stats,
        "avg_eff_ratio": avg_eff_ratio,
        "avg_stable_ratio": avg_stable_ratio,
        "energy_tol": cfg.energy_tol,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rank analysis of CeLO theta.state via SVD."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./theta.state",
        help="Path to theta.state checkpoint",
    )
    parser.add_argument(
        "--energy_tol",
        type=float,
        default=0.99,
        help="Fraction of spectral energy to retain for effective rank (default 0.99).",
    )
    parser.add_argument(
        "--min_matrix_size",
        type=int,
        default=16,
        help="Skip matrices with total size below this threshold.",
    )
    args = parser.parse_args()

    cfg = RankCfg(
        ckpt_path=args.ckpt,
        energy_tol=args.energy_tol,
        min_matrix_size=args.min_matrix_size,
    )
    estimate_theta_ranks(cfg)


if __name__ == "__main__":
    main()

