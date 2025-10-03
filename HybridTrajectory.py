#!/usr/bin/env python3
"""
Hybrid training + visualization with gradient reliability switching.

This script trains a small Transformer on enwik8 (byte-level), computes an
online gradient-reliability score R, and *switches to a zeroth-order (DeepZero-
style) two-point update* when R is low. It also trains a *shadow FO-only model*
to plot a comparison trajectory.

Outputs in --save_dir (timestamped & unique, with *_hybrid_traj suffix):
 - grad_projection_{TIMESTAMP}_hybrid_traj.png
 - grad_mag_heatmap_{TIMESTAMP}_hybrid_traj.png
 - grad_angle_hsv_{TIMESTAMP}_hybrid_traj.png
 - loss_heatmap_{TIMESTAMP}_hybrid_traj.png

NEW:
 - Marks every step where the hybrid run switched to ZO (DeepZero) with orange
   markers along the hybrid trajectory; the first switch is a larger star.
"""

import argparse
import math
import os
import random
from datetime import datetime
from urllib.request import urlretrieve

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


# --------------------------- data utils ---------------------------


def download_enwik8(target_path: str):
    url = "http://mattmahoney.net/dc/enwik8.zip"
    zip_path = target_path + ".zip"
    if os.path.exists(target_path):
        print(f"Found existing {target_path}")
        return
    print(f"Downloading enwik8 to {zip_path} (will extract to {target_path})...")
    urlretrieve(url, zip_path)
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        if "enwik8" in names:
            z.extract("enwik8", os.path.dirname(target_path) or ".")
        else:
            z.extract(names[0], os.path.dirname(target_path) or ".")
    os.remove(zip_path)
    print("Download and extract complete.")


class ByteLMDataset(Dataset):
    def __init__(self, path, seq_len, step=1):
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        self.arr = arr
        self.seq_len = seq_len
        self.step = step
        self.length = max(0, (len(arr) - seq_len) // step + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        start = i * self.step
        x = torch.from_numpy(self.arr[start : start + self.seq_len]).long()
        y = torch.from_numpy(self.arr[start + 1 : start + 1 + self.seq_len]).long()
        return x, y


# --------------------------- model ---------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,d_model)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return (
        mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    )


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        d_model=256,
        nhead=8,
        nlayers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        self.enc = nn.TransformerEncoder(block, nlayers)
        self.d_model = d_model
        self.dec = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embed(src) * math.sqrt(self.d_model)
        src = self.pos(src)
        h = self.enc(src, mask=src_mask)
        return self.dec(h)


# --------------------- parameter-space helpers ---------------------


def flatten_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flatten_grads(model) -> torch.Tensor | None:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            continue
        parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return None
    return torch.cat(parts)


def set_params_from_vector_(model, vec: torch.Tensor):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            num = p.numel()
            p.copy_(vec[idx : idx + num].view_as(p))
            idx += num


def random_orthonormal_directions(dim, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    u1 = torch.randn(dim, generator=g, device=device)
    u1 = u1 / (u1.norm() + 1e-12)
    u2 = torch.randn(dim, generator=g, device=device)
    u2 = u2 - (u2 @ u1) * u1
    u2 = u2 / (u2.norm() + 1e-12)
    return u1, u2


# --------------------- gradient reliability & ZO --------------------


class GradReliability:
    """
    Online gradient reliability estimator using low-dim random projections.
    We maintain EMA of projections to compute a proxy SNR and combine it with
    cosine similarity to a momentum direction to form a scalar score R.
    """

    def __init__(self, dim, k=32, beta_mean=0.9, beta_var=0.98, device="cpu", seed=0):
        self.dim = dim
        self.k = k
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        D = torch.empty(dim, k, device=device)
        D.bernoulli_(0.5).mul_(2.0).add_(-1.0)  # Rademacher {-1, +1}
        self.D = D / math.sqrt(dim)  # (dim,k)
        self.m = torch.zeros(k, device=device)  # EMA of projections
        self.v = torch.zeros(
            k, device=device
        )  # EMA of squared deviation (variance proxy)
        self.beta_mean = beta_mean
        self.beta_var = beta_var
        self.prev_g = None  # previous gradient vector
        self.eps = 1e-12

    @torch.no_grad()
    def update_and_score(
        self, g_vec: torch.Tensor, mom_vec: torch.Tensor | None = None
    ):
        """
        g_vec: flattened gradient (on device)
        mom_vec: flattened momentum-like vector (e.g., Adam exp_avg) (optional)
        Returns scalar R in [0,1] (heuristic).
        """
        # Projections (k,)
        p = self.D.t().mv(g_vec)  # (k,)
        # EMA of mean of projections
        self.m = self.beta_mean * self.m + (1 - self.beta_mean) * p
        # Variance proxy
        dev = p - self.m
        self.v = self.beta_var * self.v + (1 - self.beta_var) * (dev * dev)
        # SNR across probes
        snr = self.m.abs().mean() / (self.v.mean().sqrt() + self.eps)  # scalar
        snr = torch.clamp(snr, 0.0, 10.0) / 10.0  # normalize to ~[0,1]

        # Cosine similarity signals
        cos_prev = torch.tensor(0.5)  # default neutral
        if self.prev_g is not None:
            cos_prev = torch.dot(g_vec, self.prev_g) / (
                g_vec.norm() * self.prev_g.norm() + self.eps
            )
        self.prev_g = g_vec.detach().clone()

        cos_mom = None
        if mom_vec is not None and mom_vec.norm() > 0:
            cos_mom = torch.dot(g_vec, mom_vec) / (
                g_vec.norm() * mom_vec.norm() + self.eps
            )

        # Map cos in [-1,1] to [0,1]
        def cos01(x):
            return 0.5 * (x + 1.0)

        c_prev = cos01(cos_prev)
        c_mom = cos01(cos_mom) if cos_mom is not None else c_prev

        # Combine (weights can be tuned)
        R = 0.55 * snr + 0.25 * c_prev + 0.20 * c_mom
        R = torch.clamp(R, 0.0, 1.0)
        return float(R), float(snr), float(c_prev), float(c_mom)


@torch.no_grad()
def two_point_zo_step(
    model, loss_fn, data_iter, mask, sigma, lr, dirs=16, device="cpu"
):
    """
    A lightweight DeepZero-style two-point estimator step.
    Approximates gradient via k random directions with antithetic sampling,
    using ONE small minibatch from data_iter.
    """
    # Get one small batch
    try:
        x, y = next(data_iter)
    except StopIteration:
        raise RuntimeError("ZO data iterator is empty.")
    x = x.to(device)
    y = y.to(device)

    # Flatten current params
    w = flatten_params(model).to(device)
    dim = w.numel()

    # Random orthonormal-ish directions via Gram-Schmidt
    G = torch.randn(dim, dirs, device=device)
    for j in range(dirs):
        for k in range(j):
            proj = torch.dot(G[:, j], G[:, k]) * G[:, k]
            G[:, j] -= proj
        G[:, j] /= G[:, j].norm() + 1e-12

    gh = torch.zeros(dim, device=device)
    for j in range(dirs):
        d = G[:, j]
        # f(w + sigma d)
        set_params_from_vector_(model, w + sigma * d)
        logits = model(x.transpose(0, 1), src_mask=mask)
        loss_pos = loss_fn(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        ).detach()
        # f(w - sigma d)
        set_params_from_vector_(model, w - sigma * d)
        logits = model(x.transpose(0, 1), src_mask=mask)
        loss_neg = loss_fn(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        ).detach()
        # Two-point estimator contribution
        coef = (loss_pos - loss_neg) / (2.0 * sigma)
        gh += coef * d

    gh /= dirs
    # SGD-like update
    w_new = w - lr * gh
    set_params_from_vector_(model, w_new)


# --------------------------- main ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="enwik8")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--max_grad_elements", type=int, default=200000)
    ap.add_argument("--save_dir", type=str, default="out")
    ap.add_argument("--seed", type=int, default=42)

    # Field options
    ap.add_argument(
        "--field_grid", type=int, default=121, help="odd number, grid per axis"
    )
    ap.add_argument(
        "--field_radius",
        type=float,
        default=2.5,
        help="box half-width in units of param std at init",
    )
    ap.add_argument(
        "--field_batches",
        type=int,
        default=2,
        help="avg these many minibatches per grid point",
    )
    ap.add_argument(
        "--viz_batch_size", type=int, default=8, help="batch size during field eval"
    )
    ap.add_argument(
        "--eval_seq_len",
        type=int,
        default=256,
        help="shorter seq len to speed up field eval",
    )

    # Heatmap scaling
    ap.add_argument("--mag_scale", choices=["linear", "log"], default="log")
    ap.add_argument(
        "--mag_clip_p",
        type=float,
        default=99.5,
        help="percentile clip for grad magnitude",
    )

    # Trajectory capture/plotting
    ap.add_argument(
        "--traj_every",
        type=int,
        default=5,
        help="record param vector every N training steps",
    )
    ap.add_argument(
        "--traj_width", type=float, default=2.4, help="trajectory line width"
    )
    ap.add_argument(
        "--traj_alpha", type=float, default=0.95, help="trajectory line alpha [0,1]"
    )
    ap.add_argument(
        "--traj_markersize", type=float, default=9.0, help="start/end marker size"
    )

    # Plane & center choices
    ap.add_argument(
        "--plane", choices=["random", "traj_pca", "enddir"], default="traj_pca"
    )
    ap.add_argument(
        "--center_mode", choices=["init", "trained", "traj_mean"], default="trained"
    )

    # Hybrid training knobs
    ap.add_argument(
        "--reliability_k", type=int, default=32, help="# random projections for SNR"
    )
    ap.add_argument(
        "--R_threshold", type=float, default=0.45, help="below this, use ZO step"
    )
    ap.add_argument(
        "--zo_dirs", type=int, default=16, help="# directions for two-point ZO"
    )
    ap.add_argument(
        "--zo_sigma", type=float, default=1e-3, help="perturbation scale for ZO"
    )
    ap.add_argument(
        "--zo_lr",
        type=float,
        default=None,
        help="ZO step size (defaults to --lr if None)",
    )

    args = ap.parse_args()

    # Unique run-id for filenames
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.data_path):
        download_enwik8(args.data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Training loader
    train_ds = ByteLMDataset(args.data_path, seq_len=args.seq_len, step=1)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    train_iter = iter(train_dl)  # we will refresh each epoch

    # Smaller loader for field eval and ZO minibatches
    eval_ds = ByteLMDataset(args.data_path, seq_len=args.eval_seq_len, step=17)
    eval_dl = DataLoader(
        eval_ds, batch_size=args.viz_batch_size, shuffle=True, drop_last=True
    )
    eval_iter = iter(eval_dl)

    model = TransformerLM(256, args.d_model, args.nhead, args.nlayers).to(device)
    # Shadow "regular" model (FO-only path for comparison)
    model_fo = TransformerLM(256, args.d_model, args.nhead, args.nlayers).to(device)
    model_fo.load_state_dict(model.state_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer_fo = torch.optim.AdamW(model_fo.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    mask_train = generate_square_subsequent_mask(args.seq_len).to(device)

    # Initial parameter vector/stats
    w_init = flatten_params(model).to(device)
    param_std = float(w_init.std().item())
    print(f"Param std at init: {param_std:.3e}")

    # ---------------- Training + gradient PCA collection ----------------
    rng = np.random.default_rng(args.seed)
    grad_records = []
    steps_done = 0

    # record parameter trajectories (store flat vectors on CPU)
    traj_params_hybrid = [w_init.detach().cpu().clone()]  # hybrid model
    traj_params_regular = [w_init.detach().cpu().clone()]  # FO-only model

    # NEW: record parameter vectors *immediately after* each ZO step to mark them later
    zo_event_params = []  # list[Tensor] of flat params (CPU) after ZO steps
    zo_event_steps = []  # list[int] global step indices where ZO happened

    # record FLOPS
    cum_flops_hybrid = 0.0
    cum_flops_regular = 0.0

    ckpt_flops_hybrid = [0.0]
    ckpt_flops_regular = [0.0]

    # Reliability tracker
    dim = w_init.numel()
    Rtracker = GradReliability(
        dim=dim, k=args.reliability_k, device=device, seed=args.seed + 2025
    )

    # FLOPS approximation
    def fwd_flops(B, S, D, F, L):
        return L * (4 * B * S * S * D + 8 * B * S * D * D + 4 * B * S * D * F)

    C_fwd_train = fwd_flops(
        args.batch_size,
        args.seq_len,
        args.d_model,
        args.dim_feedforward if hasattr(args, "dim_feedforward") else 1024,
        args.nlayers,
    )
    C_bwd_train = 2.0 * C_fwd_train
    C_FO = C_fwd_train + C_bwd_train

    C_fwd_eval = fwd_flops(
        args.viz_batch_size,
        args.eval_seq_len,
        args.d_model,
        args.dim_feedforward if hasattr(args, "dim_feedforward") else 1024,
        args.nlayers,
    )

    pbar = tqdm(
        total=args.epochs * args.steps_per_epoch,
        desc="Training (hybrid + regular shadow)",
    )

    for epoch in range(args.epochs):
        # Refresh iters each epoch to keep data flowing
        train_iter = iter(train_dl)

        for i in range(args.steps_per_epoch):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            # ---------- 1) REGULAR (FO-only) step on shadow model ----------
            model_fo.train()
            x_fo = x.to(device)
            y_fo = y.to(device)
            src_fo = x_fo.transpose(0, 1)
            optimizer_fo.zero_grad(set_to_none=True)
            logits_fo = model_fo(src_fo, src_mask=mask_train)
            loss_fo = criterion(
                logits_fo.transpose(0, 1).reshape(-1, logits_fo.size(-1)), y_fo.view(-1)
            )
            loss_fo.backward()
            torch.nn.utils.clip_grad_norm_(model_fo.parameters(), 1.0)
            optimizer_fo.step()

            # ---------- 2) HYBRID step (use R to switch to ZO when needed) ----------
            model.train()
            x_h = x.to(device)
            y_h = y.to(device)
            src_h = x_h.transpose(0, 1)

            # Compute gradient on current minibatch (for R and possibly FO step)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src_h, src_mask=mask_train)
            loss = criterion(
                logits.transpose(0, 1).reshape(-1, logits.size(-1)), y_h.view(-1)
            )
            loss.backward()

            # Flatten current grad and Adam momentum (exp_avg) for reliability score
            g_flat = flatten_grads(model)
            mom_flat = None
            with torch.no_grad():
                m_parts = []
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        state = optimizer.state[p]
                        if "exp_avg" in state:
                            m_parts.append(state["exp_avg"].detach().reshape(-1))
                if m_parts:
                    mom_flat = torch.cat(m_parts).to(device)

            if g_flat is not None:
                arr = g_flat.detach().cpu().numpy()
                if args.max_grad_elements and arr.size > args.max_grad_elements:
                    idx = rng.choice(
                        arr.size, size=args.max_grad_elements, replace=False
                    )
                    arr = arr[idx]
                grad_records.append(arr)

            # Compute R
            R, snr, cprev, cmom = Rtracker.update_and_score(g_flat.to(device), mom_flat)
            use_zo = R < args.R_threshold

            # accumulate FLOPS
            cum_flops_hybrid += C_FO
            if use_zo:
                cum_flops_hybrid += (2 * args.zo_dirs) * C_fwd_eval
            cum_flops_regular += C_FO

            # Perform step
            if use_zo:
                # Clear grads and do ZO step with two-point estimator
                model.zero_grad(set_to_none=True)
                # Small eval-time mask for speed (reuse shorter mask)
                mask_eval = generate_square_subsequent_mask(args.eval_seq_len).to(
                    device
                )
                # Ensure eval_iter cycles
                try:
                    _ = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_dl)
                # Fresh iterator to fetch one batch inside zo_step
                eval_iter_local = iter(eval_dl)
                two_point_zo_step(
                    model,
                    criterion,
                    eval_iter_local,
                    mask_eval,
                    sigma=args.zo_sigma,
                    lr=(args.zo_lr or args.lr),
                    dirs=args.zo_dirs,
                    device=device,
                )
                # Record the param vector right after the ZO step
                zo_event_params.append(flatten_params(model).detach().cpu())
                zo_event_steps.append(steps_done + 1)  # step index *after* this update
            else:
                # FO step as usual
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            steps_done += 1

            # Record trajectories and FLOPS every N steps
            if steps_done % args.traj_every == 0:
                traj_params_hybrid.append(flatten_params(model).detach().cpu())
                traj_params_regular.append(flatten_params(model_fo).detach().cpu())

                ckpt_flops_hybrid.append(cum_flops_hybrid)
                ckpt_flops_regular.append(cum_flops_regular)

            pbar.update(1)
            if steps_done % 20 == 0:
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "R": f"{R:.3f}", "zo?": str(use_zo)}
                )

    pbar.close()

    # append final params
    traj_params_hybrid.append(flatten_params(model).detach().cpu())
    traj_params_regular.append(flatten_params(model_fo).detach().cpu())
    ckpt_flops_hybrid.append(cum_flops_hybrid)
    ckpt_flops_regular.append(cum_flops_regular)

    # ---------------- PCA of gradient snapshots ----------------
    if len(grad_records) > 0 and PCA is not None:
        X = np.stack(grad_records, axis=0)
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(
            X2[:, 0], X2[:, 1], c=np.arange(X2.shape[0]), cmap="viridis", s=24
        )
        plt.colorbar(sc, label="training step")
        plt.title("Gradient vectors projected to 2D (PCA)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.save_dir, f"grad_projection_{run_id}_hybrid_traj.png"),
            dpi=220,
        )

    # ---------------- Choose plane & center robustly (CPU SVD, NaN-safe) ----------------
    dim = w_init.numel()

    # Select center
    if args.center_mode == "init":
        center_vec = w_init
    elif args.center_mode == "trained":
        center_vec = flatten_params(model).to(device)
    else:  # 'traj_mean' over the HYBRID trajectory by default
        center_vec = torch.stack(
            [t.to(w_init.device) for t in traj_params_hybrid], dim=0
        ).mean(dim=0)

    def safe_plane_from_traj(traj_list, center_vec, seed):
        """
        Build 2D orthonormal basis (u1,u2) from trajectory with robust fallbacks.
        PCA is computed on CPU float64 and sanitized for NaN/Inf. Returns u1,u2 on center_vec.device.
        """
        dev = center_vec.device

        # Stack trajectory to CPU as float64
        T = torch.stack(traj_list, dim=0)  # (T, D), CPU
        if T.device.type != "cpu":
            T = T.cpu()
        T = T.to(torch.float64)

        # Drop rows with non-finite values
        finite_rows = torch.isfinite(T).all(dim=1)
        T = T[finite_rows]
        if T.shape[0] < 2:
            return random_orthonormal_directions(
                center_vec.numel(), device=dev, seed=seed
            )

        # Center trajectory
        T = T - T.mean(dim=0, keepdim=True)

        if T.abs().sum() < 1e-20:
            return random_orthonormal_directions(
                center_vec.numel(), device=dev, seed=seed
            )

        # SVD on CPU; float64 for stability
        try:
            Ucpu, Scpu, Vhcpu = torch.linalg.svd(T, full_matrices=False)
            u1 = Vhcpu[0].to(torch.float32)
            if Vhcpu.size(0) > 1:
                u2 = Vhcpu[1].to(torch.float32)
            else:
                u2 = torch.randn_like(u1)
        except Exception:
            # Fallback: end-direction + random orthonormal
            w_last = traj_list[-1].to(torch.float64)
            w_first = traj_list[0].to(torch.float64)
            d = w_last - w_first
            if d.norm() < 1e-12:
                return random_orthonormal_directions(
                    center_vec.numel(), device=dev, seed=seed
                )
            u1 = d.to(torch.float32)
            r = torch.randn_like(u1)
            u2 = r - (r @ u1) * u1

        # Orthonormalize and move to device
        u1 = u1 / (u1.norm() + 1e-12)
        u2 = u2 - (u2 @ u1) * u1
        u2 = u2 / (u2.norm() + 1e-12)
        return u1.to(dev), u2.to(dev)

    if args.plane == "traj_pca":
        # Use HYBRID trajectory to pick the plane (keeps the main path visible)
        u1, u2 = safe_plane_from_traj(
            traj_params_hybrid, center_vec, seed=args.seed + 123
        )
    elif args.plane == "enddir":
        w_last = traj_params_hybrid[-1].to(center_vec.device)
        d = w_last - center_vec
        if d.norm() < 1e-12:
            u1, u2 = random_orthonormal_directions(
                dim, device=center_vec.device, seed=args.seed + 123
            )
        else:
            u1 = d / (d.norm() + 1e-12)
            r = torch.randn_like(u1)
            u2 = r - (r @ u1) * u1
            u2 = u2 / (u2.norm() + 1e-12)
    else:
        u1, u2 = random_orthonormal_directions(
            dim, device=center_vec.device, seed=args.seed + 123
        )

    radius = args.field_radius * param_std

    # ---------------- Evaluate loss & gradient on the 2-D plane ----------------
    assert args.field_grid % 2 == 1, "--field_grid should be odd (e.g., 41, 81, 121)"
    lin = torch.linspace(
        -radius, radius, steps=args.field_grid, device=center_vec.device
    )
    A, B = torch.meshgrid(lin, lin, indexing="xy")

    trained_vec = flatten_params(model).to(device)  # cache to restore later

    def eval_loss_and_grad_at(w_vec):
        set_params_from_vector_(model, w_vec)
        model.zero_grad(set_to_none=True)
        total_g, total_L, batches = None, 0.0, 0
        mask_eval = generate_square_subsequent_mask(args.eval_seq_len).to(device)
        for x, y in eval_dl:
            x = x.to(device)
            y = y.to(device)
            src = x.transpose(0, 1)
            logits = model(src, src_mask=mask_eval)
            loss = criterion(
                logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
            )
            loss.backward()
            g = flatten_grads(model)
            if g is not None:
                total_g = g if total_g is None else (total_g + g)
                total_L += float(loss.item())
                batches += 1
            model.zero_grad(set_to_none=True)
            if batches >= args.field_batches:
                break
        if batches == 0:
            return 0.0, torch.zeros_like(center_vec)
        return total_L / batches, (total_g / batches).detach()

    U = torch.zeros_like(A)  # descent x
    V = torch.zeros_like(B)  # descent y
    L = torch.zeros_like(A)  # loss

    print("Computing gradient field & loss heatmaps...")

    for i in tqdm(range(args.field_grid)):
        for j in range(args.field_grid):
            a, b = A[i, j].item(), B[i, j].item()
            w = center_vec + a * u1 + b * u2
            torch.set_grad_enabled(True)
            loss_ij, g = eval_loss_and_grad_at(w)
            torch.set_grad_enabled(False)
            gx = torch.dot(g.to(device), u1).item()
            gy = torch.dot(g.to(device), u2).item()
            U[i, j] = -gx
            V[i, j] = -gy
            L[i, j] = loss_ij

    set_params_from_vector_(model, trained_vec)  # restore trained weights

    # ---------------- Heatmaps + literal trajectories (two) + ZO markers ----------------
    a = A.detach().cpu().numpy()
    b = B.detach().cpu().numpy()
    u = U.detach().cpu().numpy()
    v = V.detach().cpu().numpy()
    mag = np.hypot(u, v)
    ang = np.arctan2(v, u)  # [-pi, pi]

    # Project trajectories and ZO-event points onto (u1,u2)
    u1_cpu = u1.detach().cpu()
    u2_cpu = u2.detach().cpu()
    center_cpu = center_vec.detach().cpu()

    def project_list(vec_list):
        if not vec_list:
            return np.zeros((0, 2), dtype=np.float64)
        out = []
        for w_vec in vec_list:
            d = w_vec.detach().cpu() - center_cpu
            out.append([float(torch.dot(d, u1_cpu)), float(torch.dot(d, u2_cpu))])
        return np.array(out, dtype=np.float64)

    traj_h = project_list(traj_params_hybrid)
    traj_r = project_list(traj_params_regular)
    zo_pts = project_list(zo_event_params)  # points after each ZO step

    # Convert FLOP checkpoints to numpy arrays aligned with traj points
    fl_h = np.asarray(ckpt_flops_hybrid, dtype=np.float64)
    fl_r = np.asarray(ckpt_flops_regular, dtype=np.float64)

    # Safety: lengths should match # of points in traj_h / traj_r
    fl_h = fl_h[: len(traj_h)]
    fl_r = fl_r[: len(traj_r)] if traj_r.size else np.zeros(0, dtype=np.float64)

    # Shared normalization so colors are comparable across runs
    max_total_flops = max(
        float(fl_h[-1]) if fl_h.size else 0.0, float(fl_r[-1]) if fl_r.size else 0.0
    )
    norm_flops = mcolors.Normalize(vmin=0.0, vmax=max_total_flops)

    def _flop_units(x: float):
        if x >= 1e12:
            return 1e12, "TFLOPs"
        if x >= 1e9:
            return 1e9, "GFLOPs"
        if x >= 1e6:
            return 1e6, "MFLOPs"
        return 1.0, "FLOPs"

    scale_flops, unit_flops = _flop_units(
        max_total_flops if max_total_flops > 0 else 1.0
    )

    def add_colored_trajectory(ax, xy, flops, label):
        """
        Draws a line whose segments are colored by cumulative FLOP-proxy.
        Colors correspond to the FLOPs *at the segment end*.
        """
        if len(xy) < 2:
            return None
        segs = np.stack([xy[:-1], xy[1:]], axis=1)  # (N-1, 2, 2)
        lc = LineCollection(
            segs,
            cmap="viridis",
            norm=norm_flops,
            linewidth=args.traj_width,
            alpha=args.traj_alpha,
            zorder=6,
        )
        lc.set_array(flops[1:])  # color by cumulative FLOPs at end of each segment
        ax.add_collection(lc)
        # Start/end markers
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            s=args.traj_markersize**2,
            c="#33dd33",
            edgecolors="black",
            linewidths=0.9,
            zorder=7,
        )
        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            s=args.traj_markersize**2,
            c="#ff3333",
            edgecolors="black",
            linewidths=0.9,
            zorder=7,
            label=label,
        )
        return lc


    def square_extent(extent):
        """Return a squared extent by padding the shorter dimension symmetrically."""
        xmin, xmax, ymin, ymax = extent
        dx = xmax - xmin
        dy = ymax - ymin
        if dx == dy:
            return extent
        if dx > dy:
            pad = (dx - dy) / 2.0
            return (xmin, xmax, ymin - pad, ymax + pad)
        else:
            pad = (dy - dx) / 2.0
            return (xmin - pad, xmax + pad, ymin, ymax)

    # Base extent from grid, then expand to include both trajectories and ZO points
    extent = [a.min(), a.max(), b.min(), b.max()]
    if (traj_h.size > 0) or (traj_r.size > 0) or (zo_pts.size > 0):
        tx = np.concatenate(
            [traj_h[:, 0]]
            + ([traj_r[:, 0]] if traj_r.size else [])
            + ([zo_pts[:, 0]] if zo_pts.size else [])
        )
        ty = np.concatenate(
            [traj_h[:, 1]]
            + ([traj_r[:, 1]] if traj_r.size else [])
            + ([zo_pts[:, 1]] if zo_pts.size else [])
        )
        pad_x = 0.05 * (extent[1] - extent[0] + 1e-12)
        pad_y = 0.05 * (extent[3] - extent[2] + 1e-12)
        extent[0] = min(extent[0], tx.min()) - pad_x
        extent[1] = max(extent[1], tx.max()) + pad_x
        extent[2] = min(extent[2], ty.min()) - pad_y
        extent[3] = max(extent[3], ty.max()) + pad_y

    extent = square_extent(extent)

    # robust clip & scaling for magnitude
    clip_hi = np.percentile(mag, 99.5)
    mag_clipped = np.clip(mag, 0, clip_hi)
    if args.mag_scale == "log":
        mag_vis = np.log10(mag_clipped + 1e-12)
        mag_vis -= mag_vis.min()
        mag_vis /= mag_vis.max() + 1e-12
        mag_label = "‖projected grad‖ (log10, robust)"
    else:
        mag_vis = mag_clipped / (mag_clipped.max() + 1e-12)
        mag_label = "‖projected grad‖ (robust)"

    def draw_both_trajs_and_zo(ax):
        """
        OLD METHOD, before FLOPS
        """

        # HYBRID in white (with halo), REGULAR in cyan (with halo)
        # Hybrid
        (line_h,) = ax.plot(
            traj_h[:, 0],
            traj_h[:, 1],
            color="white",
            linewidth=args.traj_width,
            alpha=args.traj_alpha,
            zorder=6,
            label="Hybrid",
        )
        line_h.set_path_effects(
            [
                patheffects.Stroke(
                    linewidth=args.traj_width + 1.6, foreground="black", alpha=0.75
                ),
                patheffects.Normal(),
            ]
        )
        ax.scatter(
            traj_h[0, 0],
            traj_h[0, 1],
            s=args.traj_markersize**2,
            c="#33dd33",
            edgecolors="black",
            linewidths=0.9,
            zorder=7,
        )
        ax.scatter(
            traj_h[-1, 0],
            traj_h[-1, 1],
            s=args.traj_markersize**2,
            c="#ff3333",
            edgecolors="black",
            linewidths=0.9,
            zorder=7,
        )
        # Regular
        if traj_r.size:
            (line_r,) = ax.plot(
                traj_r[:, 0],
                traj_r[:, 1],
                color="#66ddff",
                linewidth=args.traj_width * 0.9,
                alpha=0.85,
                zorder=5,
                label="Regular (FO-only)",
            )
            line_r.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=args.traj_width * 0.9 + 1.4,
                        foreground="black",
                        alpha=0.55,
                    ),
                    patheffects.Normal(),
                ]
            )
        # ZO switch markers (orange)
        if zo_pts.size:
            ax.scatter(
                zo_pts[:, 0],
                zo_pts[:, 1],
                s=42,
                marker="x",
                c="#ffa500",
                edgecolors="black",
                linewidths=0.7,
                zorder=8,
                label="ZO step",
            )
            # Highlight first switch (star, bigger)
            ax.scatter(
                zo_pts[0, 0],
                zo_pts[0, 1],
                s=120,
                marker="*",
                c="#ffa500",
                edgecolors="black",
                linewidths=0.8,
                zorder=9,
                label="First ZO switch",
            )
        ax.legend(loc="upper right", frameon=True)

    # 1) Gradient magnitude heatmap + trajectories + ZO markers
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(mag_vis, origin="lower", extent=extent, aspect="equal", cmap="magma")
    cbar = fig.colorbar(im)
    cbar.set_label(mag_label)
    draw_both_trajs_and_zo(ax)
    ax.set_title(
        f"Gradient Magnitude + Trajectories  (plane: {args.plane}, center: {args.center_mode})"
    )
    ax.set_xlabel("a (along u1)")
    ax.set_ylabel("b (along u2)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.save_dir, f"grad_mag_heatmap_{run_id}_hybrid_traj.png"),
        dpi=220,
    )

    # 1.B) Gradient magnitude heatmap + trajectories + ZO markers + FLOPS (with dual colorbars)
    fig, ax = plt.subplots(
        figsize=(8.5, 7), constrained_layout=True
    )  # a bit wider helps


    # Heatmap + its own colorbar (right)
    im = ax.imshow(mag_vis, origin="lower", extent=extent, aspect="equal", cmap="magma")
    cbar_mag = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, location="right")
    cbar_mag.set_label(mag_label)

    # Draw FLOP-colored trajectories
    lc_main = None
    if traj_h.size:
        lc_main = add_colored_trajectory(ax, traj_h, fl_h, label="Hybrid (FLOPs)")
    if traj_r.size:
        _ = add_colored_trajectory(ax, traj_r, fl_r, label="Regular FO-only (FLOPs)")

    # ZO markers
    if zo_pts.size:
        ax.scatter(
            zo_pts[:, 0],
            zo_pts[:, 1],
            s=42,
            marker="x",
            c="#ffa500",
            edgecolors="black",
            linewidths=0.7,
            zorder=8,
            label="ZO step",
        )
        ax.scatter(
            zo_pts[0, 0],
            zo_pts[0, 1],
            s=120,
            marker="*",
            c="#ffa500",
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
            label="First ZO switch",
        )

    # FLOPs colorbar on the LEFT (no axes_grid1; constrained_layout handles spacing)
    if lc_main is not None:
        cbar_flops = fig.colorbar(
            lc_main, ax=ax, location="left", fraction=0.046, pad=0.12
        )
        cbar_flops.ax.yaxis.set_label_position("left")
        cbar_flops.ax.yaxis.tick_left()
        fmt = FuncFormatter(lambda val, pos: f"{val/scale_flops:.1f}")
        cbar_flops.formatter = fmt
        cbar_flops.update_ticks()
        cbar_flops.set_label(f"Cumulative FLOPs ({unit_flops})")

    ax.legend(loc="upper right", frameon=True)
    ax.set_title(
        f"Gradient Magnitude + Trajectories (FLOP-colored)\n"
        f"(plane: {args.plane}, center: {args.center_mode})"
    )
    ax.set_xlabel("a (along u1)")
    ax.set_ylabel("b (along u2)")
    ax.set_box_aspect(1)

    # No tight_layout() here — constrained_layout already did spacing
    fig.savefig(
        os.path.join(args.save_dir, f"grad_mag_heatmap_{run_id}_hybrid_traj_FLOPS.png"),
        dpi=220,
    )

    # 2) HSV image: hue=direction, value=magnitude + trajectories + ZO markers
    h = (ang + np.pi) / (2 * np.pi)  # [0,1] hue
    s_img = np.ones_like(h)  # full saturation
    v_img = mag_vis  # normalized magnitude as value
    rgb = mcolors.hsv_to_rgb(np.stack([h, s_img, v_img], axis=-1))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin="lower", extent=extent, aspect="equal")
    draw_both_trajs_and_zo(ax)
    ax.set_title(
        f"Gradient Direction (hue) + Magnitude (value) + Trajectories\n(plane: {args.plane}, center: {args.center_mode})"
    )
    ax.set_xlabel("a (along u1)")
    ax.set_ylabel("b (along u2)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.save_dir, f"grad_angle_hsv_{run_id}_hybrid_traj.png"), dpi=220
    )

    # 3) Loss heatmap + trajectories + ZO markers
    loss = L.detach().cpu().numpy()
    Lhi, Llo = np.percentile(loss, 99.0), np.percentile(loss, 1.0)
    Lvis = np.clip(loss, Llo, Lhi)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(Lvis, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    fig.colorbar(im, label="Loss (robust scaled)")
    draw_both_trajs_and_zo(ax)
    ax.set_title(
        f"Loss Landscape + Trajectories  (plane: {args.plane}, center: {args.center_mode})"
    )
    ax.set_xlabel("a (along u1)")
    ax.set_ylabel("b (along u2)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.save_dir, f"loss_heatmap_{run_id}_hybrid_traj.png"), dpi=220
    )

    # Helpful console printout
    if zo_event_steps:
        print(
            f"ZO steps occurred at global step indices (post-update): {zo_event_steps}"
        )
        print(f"First ZO switch at step {zo_event_steps[0]}")
    else:
        print("No ZO steps occurred (R never fell below threshold).")

    print(f"Saved figures with timestamp {run_id} to", args.save_dir)


if __name__ == "__main__":
    main()
