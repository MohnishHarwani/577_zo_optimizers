#!/usr/bin/env python3
"""
EMA-smoothed plotter for CeLO quantization CSV.

Input CSV columns (as produced by quantization_study.py):
  dataset, run, step, optimizer, train_loss, val_loss

Produces:
  - Smoothed mean curves (EMA on mean across runs)
  - Shaded band (either raw std or EMA-smoothed std)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- CONFIG ----------------------------------

@dataclass
class PlotCfg:
    csv_path: str = "celo_quant_transformer_steps.csv"
    out_path: str = "celo_quant_transformer_steps_EMA.pdf"

    # Which series to plot: "train_loss" or "val_loss"
    y_col: str = "train_loss"

    # EMA smoothing:
    # Option A: specify alpha directly (0<alpha<=1). Larger alpha = less smoothing.
    ema_alpha: float = 0.05

    # Option B (alternative): specify "half-life" in steps; alpha computed from it.
    # If half_life_steps is not None, it overrides ema_alpha.
    half_life_steps: int | None = None

    # Apply EMA to the std band too (looks nicer, but slightly less "honest" about variance).
    smooth_std: bool = False

    # If True, ignore step==0 rows (train_loss is NaN at step 0 in your writer).
    drop_step0: bool = True

    title: str = "TransformerLM_LM1B: CeLO quant sweep (EMA-smoothed mean ± std)"
    xlabel: str = "Step"
    ylabel: str = "Loss"


CFG = PlotCfg()


# --------------------------- EMA UTILS -------------------------------

def ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    """Simple EMA for 1D array. NaNs are skipped (carry previous EMA)."""
    out = np.empty_like(x, dtype=np.float64)
    prev = np.nan
    for i, v in enumerate(x.astype(np.float64)):
        if np.isnan(v):
            out[i] = prev
            continue
        if np.isnan(prev):
            prev = v
        else:
            prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def alpha_from_half_life(half_life_steps: int) -> float:
    # Half-life H: after H steps, weight decays to 0.5 => (1-alpha)^H = 0.5
    return 1.0 - (0.5 ** (1.0 / float(half_life_steps)))


# --------------------------- MAIN ------------------------------------

def main():
    cfg = CFG
    df = pd.read_csv(cfg.csv_path)

    required = {"dataset", "run", "step", "optimizer", "train_loss", "val_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if cfg.drop_step0:
        df = df[df["step"] > 0].copy()

    if cfg.y_col not in ("train_loss", "val_loss"):
        raise ValueError("y_col must be 'train_loss' or 'val_loss'")

    # Compute mean/std across runs per step for each optimizer
    g = (
        df.groupby(["optimizer", "step"], as_index=False)
          .agg(mean_y=(cfg.y_col, "mean"), std_y=(cfg.y_col, "std"))
    )
    g["std_y"] = g["std_y"].fillna(0.0)

    # Choose EMA alpha
    alpha = cfg.ema_alpha
    if cfg.half_life_steps is not None:
        alpha = alpha_from_half_life(cfg.half_life_steps)

    plt.figure(figsize=(8, 5))
    plt.title(cfg.title)

    for opt_name in sorted(g["optimizer"].unique()):
        sub = g[g["optimizer"] == opt_name].sort_values("step")

        x = sub["step"].to_numpy()
        y = sub["mean_y"].to_numpy()
        s = sub["std_y"].to_numpy()

        y_ema = ema_1d(y, alpha=alpha)
        if cfg.smooth_std:
            s_use = ema_1d(s, alpha=alpha)
        else:
            s_use = s

        plt.plot(x, y_ema, label=opt_name)
        plt.fill_between(x, y_ema - s_use, y_ema + s_use, alpha=0.2)

    plt.xlabel(cfg.xlabel)
    plt.ylabel(cfg.ylabel if cfg.ylabel else cfg.y_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.out_path, dpi=300)
    print(f"[OK] Saved {cfg.out_path} (alpha={alpha:.6f}, smooth_std={cfg.smooth_std})")


if __name__ == "__main__":
    main()


