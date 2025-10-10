#!/usr/bin/env python3
"""
AdamW vs AdaSign-Lite vs LSTM-LO vs PyLO (VeLO, AdafacLO) on enwik8 (byte-level).

Outputs in --save_dir:
  - loss_vs_steps_{TIMESTAMP}.png
  - loss_vs_flops_{TIMESTAMP}.png
  - logs_{TIMESTAMP}.npz
"""

import argparse
import math
import os
import random
from datetime import datetime
from urllib.request import urlretrieve
from typing_extensions import Optional, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------- optional PyLO ---------------------------
PYLO_AVAILABLE = True
try:
    from pylo.optim import AdafacLO_naive, MuLO_naive, VeLO
except Exception:
    PYLO_AVAILABLE = False

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
        self.arr, self.seq_len, self.step = arr, seq_len, step
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
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, nhead=8, nlayers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.enc = nn.TransformerEncoder(block, nlayers)
        self.d_model = d_model
        self.dec = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embed(src) * math.sqrt(self.d_model)
        src = self.pos(src)
        h = self.enc(src, mask=src_mask)
        return self.dec(h)

class CeLOLite(nn.Module):
    """
    Minimal CeLO-style learned optimizer (single global scheduler + per-parameter MLP)
    with numerical safety (clamps & clipping) to prevent NaNs.
    """

    def __init__(
        self,
        hidden_sched: int = 32,
        hidden_rule: int = 32,
        alpha: float = 0.1,         # more conservative default
        lambda1: float = 1.0,
        lambda2: float = 0.1,       # more conservative default
        ema_loss_beta: float = 0.95,
        ema_grad2_beta: float = 0.99,
        eps: float = 1e-8,
        # safety knobs
        o_clip: float = 6.0,        # clamp pre-exp scheduler output
        mag_clip: float = 6.0,      # clamp pre-exp magnitude
        update_clip_ratio: float = 0.1,  # ||Δp|| <= ratio * ||p||
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ema_loss_beta = ema_loss_beta
        self.ema_grad2_beta = ema_grad2_beta
        self.eps = eps
        self.o_clip = o_clip
        self.mag_clip = mag_clip
        self.update_clip_ratio = update_clip_ratio

        # --- Learned scheduler (global) ---
        # features: [log1p(step), norm_loss, dmean, rms_g]
        self.sched_in = 4
        self.sched_cell = nn.LSTMCell(self.sched_in, hidden_sched)
        self.sched_head = nn.Linear(hidden_sched, 1)
        self.sched_h: Optional[torch.Tensor] = None
        self.sched_c: Optional[torch.Tensor] = None

        # --- Learned update rule (per-parameter; shared weights) ---
        # features per param: [g, m_ema, rms_g] (simple stats)
        self.rule_in = 3
        self.rule = nn.Sequential(
            nn.Linear(self.rule_in, hidden_rule),
            nn.Tanh(),
            nn.Linear(hidden_rule, hidden_rule),
            nn.Tanh(),
        )
        self.rule_dir = nn.Linear(hidden_rule, 1)  # d (direction, later tanh)
        self.rule_mag = nn.Linear(hidden_rule, 1)  # m (magnitude, later exp)

        # persistent per-tensor state
        self._state: Dict[torch.nn.Parameter, Dict[str, torch.Tensor]] = {}

        if device is not None:
            self.to(device)

    def reset_state(self):
        self._state.clear()
        self.sched_h = None
        self.sched_c = None
        if hasattr(self, "_ema_loss"):
            delattr(self, "_ema_loss")

    def _get_tensor_state(self, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        st = self._state.get(p)
        if st is None or any(t.device != p.device for t in st.values()):
            st = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            self._state[p] = st
        return st

    @torch.no_grad()
    def step_inplace(self, params, loss_value: float, step_index: int):
        # ---- 1) Update accumulators & gather global feats
        device = None
        gnorm_sum, g2mean_sum, n_tensors = 0.0, 0.0, 0
        for p in params:
            if p.grad is None:
                continue
            device = p.device
            st = self._get_tensor_state(p)
            g = p.grad
            st["m"].mul_(0.9).add_(g, alpha=0.1)
            st["v"].mul_(self.ema_grad2_beta).addcmul_(g, g, value=(1.0 - self.ema_grad2_beta))
            gnorm_sum += float(g.norm().item())
            g2mean_sum += float(g.pow(2).mean().item())
            n_tensors += 1
        if device is None:
            return  # nothing to do

        # loss EMA (scalar)
        if not hasattr(self, "_ema_loss"):
            self._ema_loss = torch.tensor(float(loss_value), device=device)
        self._ema_loss = self._ema_loss * self.ema_loss_beta + (1.0 - self.ema_loss_beta) * float(loss_value)

        # ---- 2) Scheduler features -> (1,4)
        prog  = torch.tensor(math.log1p(step_index), device=device)
        lfeat = torch.tensor(math.log1p(float(self._ema_loss)), device=device)
        dmean = torch.tensor(gnorm_sum / max(n_tensors, 1) + 1e-12, device=device)
        rmsg  = torch.tensor(math.sqrt(max(g2mean_sum, 1e-16)), device=device)
        xs = torch.stack((prog, lfeat, dmean, rmsg)).unsqueeze(0)  # (1,4)

        # ---- 3) Scheduler forward with clamp before exp
        if (self.sched_h is None or self.sched_c is None or self.sched_h.device != device):
            H = self.sched_cell.hidden_size
            self.sched_h = torch.zeros(1, H, device=device)
            self.sched_c = torch.zeros(1, H, device=device)

        self.sched_h, self.sched_c = self.sched_cell(xs, (self.sched_h, self.sched_c))
        o_t = self.sched_head(self.sched_h).clamp(-self.o_clip, self.o_clip)  # (1,1)
        eta_t = float(self.alpha) * torch.exp(o_t).squeeze()  # scalar tensor in (0, alpha*e^o_clip]

        # ---- 4) Per-parameter updates (vectorized per tensor)
        for p in params:
            if p.grad is None:
                continue
            st = self._get_tensor_state(p)
            g, m, v = p.grad, st["m"], st["v"]
            rms = v.sqrt() + self.eps

            # per-coordinate features: [g, m, rms]
            feats = torch.stack([g, m, rms], dim=-1).reshape(-1, self.rule_in)  # (N,3)
            h = self.rule(feats)
            d_raw = self.rule_dir(h).squeeze(-1)     # (N,)
            mag   = self.rule_mag(h).squeeze(-1)     # (N,)

            # safety: bound direction and magnitude
            d   = torch.tanh(d_raw)                  # in [-1, 1]
            mag = mag.clamp(-self.mag_clip, self.mag_clip)
            scale_mag = torch.exp(self.lambda2 * mag)    # <= exp(lambda2*mag_clip)

            update_flat = self.lambda1 * d * scale_mag
            update = update_flat.view_as(p)

            # multiply by global ||p||_2 and scheduler
            pn = p.norm() + self.eps
            update = eta_t * update * pn

            # clip update norm relative to parameter norm
            u_max = self.update_clip_ratio * pn
            u_norm = update.norm() + self.eps
            if u_norm > u_max:
                update.mul_(u_max / u_norm)

            # final guard: zap non-finite updates
            if not torch.isfinite(update).all():
                update.zero_()

            p.add_(-update)
# --------------------------- FLOPs proxy ---------------------------

def fwd_flops(B, S, D, F, L):
    return L * (4 * B * S * S * D + 8 * B * S * D * D + 4 * B * S * D * F)

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# --------------------------- LSTM Learned Optimizer ---------------------------

class LSTMLOptimizer(nn.Module):
    """Coordinate-wise LSTM learned optimizer (shared weights)."""

    def __init__(self, feature_rms=True, hidden_size=32, eps=1e-8):
        super().__init__()
        self.feature_rms, self.eps = feature_rms, eps
        in_dim = 2 + (1 if feature_rms else 0)
        self.cell = nn.LSTMCell(in_dim, hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.scale_bias = nn.Parameter(torch.tensor(0.0))
        self._state = {}  # p -> (h, c, rms)

    def reset_state(self):
        self._state.clear()

    @torch.no_grad()
    def _get_state_tensors(self, p, H, device):
        h, c, rms = self._state.get(p, (None, None, None))
        n = p.numel()
        if h is None or h.numel() != n * H:
            h = torch.zeros(n, H, device=device)
            c = torch.zeros(n, H, device=device)
        if self.feature_rms:
            if rms is None or rms.numel() != n:
                rms = torch.zeros(n, device=device)
        else:
            rms = None
        return h, c, rms

    def _set_state_tensors(self, p, h, c, rms):
        self._state[p] = (h, c, rms)

    def _features_from_grad(self, g_flat, rms_flat):
        abs_g, sgn_g = g_flat.abs(), torch.sign(g_flat)
        return (torch.stack([abs_g, sgn_g, rms_flat.clamp_min(1e-12)], dim=-1)
                if self.feature_rms else torch.stack([abs_g, sgn_g], dim=-1))

    def propose_update(self, params, lr=3e-4, beta_rms=0.99):
        updates = []
        for p in params:
            if p.grad is None:
                updates.append(None)
                continue
            g = p.grad
            device = p.device
            H = self.cell.hidden_size
            g_flat = g.detach().reshape(-1)
            h, c, rms = self._get_state_tensors(p, H, device)
            if self.feature_rms:
                if rms is None:
                    rms = torch.zeros_like(g_flat)
                rms = beta_rms * rms + (1.0 - beta_rms) * g_flat.pow(2)
                denom = rms.sqrt() + self.eps
            else:
                denom = g_flat.new_ones(g_flat.shape)
            feats = self._features_from_grad(g_flat, rms if rms is not None else g_flat.new_zeros(g_flat.shape))
            h, c = self.cell(feats, (h, c))
            s = self.head(h).squeeze(-1)
            scale = torch.nn.functional.softplus(s + self.scale_bias) + 1e-6
            upd_flat = -lr * scale * (g_flat / denom)
            updates.append(upd_flat.view_as(p))
            self._set_state_tensors(p, h.detach(), c.detach(), rms.detach() if rms is not None else None)
        return updates

    @torch.no_grad()
    def step_inplace(self, params, lr=3e-4, beta_rms=0.99):
        ups = self.propose_update(params, lr=lr, beta_rms=beta_rms)
        for p, u in zip(params, ups):
            if u is not None:
                p.add_(u)
        return ups

# --------------------------- AdaSign-Lite ---------------------------

class AdaSignLite(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999, gamma=1e-3, eps=1e-8, eps2=1e-12, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, gamma=gamma, eps=eps, eps2=eps2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta1, beta2, gamma, eps, eps2, wd = (
                group["lr"], group["beta1"], group["beta2"], group["gamma"], group["eps"], group["eps2"], group["weight_decay"]
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["s"] = torch.zeros_like(p)
                m, v, s = st["m"], st["v"], st["s"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = v.sqrt() + eps
                align = (g * m) / (denom + eps2)
                s.add_(align, alpha=gamma)
                step_scale = torch.exp(torch.clamp(s, -10, 10))
                p.add_(-(lr * step_scale) * (g / (denom + eps2)))
        return None

# --------------------------- training helpers ---------------------------

def train_one_celo(model, celo: CeLOLite, train_dl, steps, device, seq_len, d_model, dim_ff, nlayers, log_every=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    # FLOPs proxy (same as others)
    B, S, D, F, L = train_dl.batch_size, seq_len, d_model, dim_ff, nlayers
    C_fwd = fwd_flops(B, S, D, F, L)
    C_bwd = 2.0 * C_fwd
    C_step = C_fwd + C_bwd
    cum_flops = 0.0

    losses, flops = [], []
    it = iter(train_dl)
    celo.reset_state()

    for t in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        src = x.transpose(0, 1)

        # regular forward/backward
        for p in model.parameters():
            p.grad = None
        logits = model(src, src_mask=mask)
        loss = criterion(logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # CeLO update (in-place)
        celo.step_inplace(list(model.parameters()), loss_value=float(loss.item()), step_index=t)

        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")

    return np.array(losses, float), np.array(flops, float)

def train_one_torchopt(model, optimizer, train_dl, steps, device, seq_len, d_model, dim_ff, nlayers, log_every=10):
    """torch.optim-style (AdamW, AdaSign-Lite)"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)
    B, S, D, F, L = train_dl.batch_size, seq_len, d_model, dim_ff, nlayers
    C_fwd, C_step = fwd_flops(B, S, D, F, L), 3.0 * fwd_flops(B, S, D, F, L)
    cum_flops, losses, flops = 0.0, [], []
    it = iter(train_dl)
    for t in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, src_mask=mask)
        loss = criterion(logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")
    return np.array(losses, float), np.array(flops, float)

def train_one_lstm(model, learned_opt: LSTMLOptimizer, base_lr, train_dl, steps, device, seq_len, d_model, dim_ff, nlayers, log_every=10):
    """Functional LSTM-LO"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)
    # ensure LO is on right device
    if any(p.requires_grad for p in learned_opt.parameters()):
        if next(learned_opt.parameters()).device != device:
            learned_opt.to(device)
    B, S, D, F, L = train_dl.batch_size, seq_len, d_model, dim_ff, nlayers
    C_step = 3.0 * fwd_flops(B, S, D, F, L)
    cum_flops, losses, flops = 0.0, [], []
    it = iter(train_dl)
    learned_opt.reset_state()
    for t in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0, 1)
        for p in model.parameters():
            p.grad = None
        logits = model(src, src_mask=mask)
        loss = criterion(logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learned_opt.step_inplace(list(model.parameters()), lr=base_lr, beta_rms=0.99)
        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")
    return np.array(losses, float), np.array(flops, float)

def train_one_pylo(model, optimizer, train_dl, steps, device, seq_len, d_model, dim_ff, nlayers, log_every=10):
    """
    PyLO optimizers (VeLO, AdafacLO_naive/MuLO_naive): use .step(loss)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)
    B, S, D, F, L = train_dl.batch_size, seq_len, d_model, dim_ff, nlayers
    C_step = 3.0 * fwd_flops(B, S, D, F, L)
    cum_flops, losses, flops = 0.0, [], []
    it = iter(train_dl)
    for t in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, src_mask=mask)
        loss = criterion(logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(loss)  # <-- PyLO requires the loss
        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")
    return np.array(losses, float), np.array(flops, float)

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="enwik8")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--dim_feedforward", type=int, default=1024)
    ap.add_argument("--save_dir", type=str, default="out_compare")
    ap.add_argument("--seed", type=int, default=42)
    # AdaSign-Lite hypers
    ap.add_argument("--asl_beta1", type=float, default=0.9)
    ap.add_argument("--asl_beta2", type=float, default=0.999)
    ap.add_argument("--asl_gamma", type=float, default=1e-3)
    ap.add_argument("--asl_eps", type=float, default=1e-8)
    ap.add_argument("--asl_eps2", type=float, default=1e-12)
    # LSTM-LO hypers
    ap.add_argument("--lstm_hidden", type=int, default=32)
    ap.add_argument("--lstm_feature_rms", action="store_true", default=True)
    # Which PyLO optimizers to run
    ap.add_argument("--use_velo", action="store_true", default=True)
    ap.add_argument("--use_adafaclo", action="store_true", default=True)
    ap.add_argument("--use_mulo", action="store_true", default=False)  # off by default; requires MuP shapes

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.data_path):
        download_enwik8(args.data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Batch size:", args.batch_size)

    # Data
    train_ds = ByteLMDataset(args.data_path, seq_len=args.seq_len, step=1)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Base model
    def make_model():
        return TransformerLM(256, args.d_model, args.nhead, args.nlayers, dim_feedforward=args.dim_feedforward).to(device)

    # AdamW
    model_adam = make_model()
    model_asl = make_model()
    model_asl.load_state_dict(model_adam.state_dict())
    model_lstm = make_model()
    model_lstm.load_state_dict(model_adam.state_dict())
    # PyLO mirrors (if available)
    model_velo, model_adafaclo, model_mulo = None, None, None
    if PYLO_AVAILABLE and args.use_velo:
        model_velo = make_model()
        model_velo.load_state_dict(model_adam.state_dict())
    if PYLO_AVAILABLE and args.use_adafaclo:
        model_adafaclo = make_model()
        model_adafaclo.load_state_dict(model_adam.state_dict())
    if PYLO_AVAILABLE and args.use_mulo:
        model_mulo = make_model()
        model_mulo.load_state_dict(model_adam.state_dict())

    total_steps = args.epochs * args.steps_per_epoch
    print(f"Training {total_steps} steps per run...")

    # Optimizers
    opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=args.lr)
    opt_asl = AdaSignLite(
        model_asl.parameters(),
        lr=args.lr,
        beta1=args.asl_beta1,
        beta2=args.asl_beta2,
        gamma=args.asl_gamma,
        eps=args.asl_eps,
        eps2=args.asl_eps2,
    )
    learned_opt = LSTMLOptimizer(feature_rms=args.lstm_feature_rms, hidden_size=args.lstm_hidden).to(device)

    # Logs dict
    curves = {}

    # AdamW
    print("\n=== AdamW ===")
    losses_adam, flops_adam = train_one_torchopt(
        model_adam, opt_adam, train_dl, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
    )
    curves["AdamW"] = (losses_adam, flops_adam)

    # AdaSign-Lite
    print("\n=== AdaSign-Lite ===")
    train_dl2 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    losses_asl, flops_asl = train_one_torchopt(
        model_asl, opt_asl, train_dl2, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
    )
    curves["AdaSign-Lite"] = (losses_asl, flops_asl)

    # LSTM-LO
    print("\n=== LSTM Learned Optimizer ===")
    train_dl3 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    losses_lstm, flops_lstm = train_one_lstm(
        model_lstm, learned_opt, args.lr, train_dl3, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
    )
    curves["LSTM-LO"] = (losses_lstm, flops_lstm)

    # PyLO: VeLO
    if PYLO_AVAILABLE and model_velo is not None:
        print("\n=== PyLO VeLO ===")
        opt_velo = VeLO(model_velo.parameters())  # default cfg; PyLO expects .step(loss)
        train_dl4 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        losses_velo, flops_velo = train_one_pylo(
            model_velo, opt_velo, train_dl4, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
        )
        curves["VeLO"] = (losses_velo, flops_velo)
    elif not PYLO_AVAILABLE and args.use_velo:
        print(">> PyLO not installed; skipping VeLO.")

    # PyLO: AdafacLO (naive)
    if PYLO_AVAILABLE and model_adafaclo is not None:
        print("\n=== PyLO AdafacLO (naive) ===")
        opt_af = AdafacLO_naive(model_adafaclo.parameters())
        train_dl5 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        losses_af, flops_af = train_one_pylo(
            model_adafaclo, opt_af, train_dl5, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
        )
        curves["AdafacLO"] = (losses_af, flops_af)
    elif not PYLO_AVAILABLE and args.use_adafaclo:
        print(">> PyLO not installed; skipping AdafacLO.")

    # PyLO: MuLO (μP wrapper) — requires MuP base shapes; left off by default
    if PYLO_AVAILABLE and model_mulo is not None:
        print("\n=== PyLO MuLO (μP wrapper over AdafacLO) ===")
        try:
            opt_mulo = MuLO_naive(model_mulo.parameters())
            train_dl6 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            losses_mulo, flops_mulo = train_one_pylo(
                model_mulo, opt_mulo, train_dl6, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
            )
            curves["MuLO"] = (losses_mulo, flops_mulo)
        except Exception as e:
            print(f">> MuLO run skipped (needs MuP shapes): {e}")

    # --- models (after model_lstm) ---
    model_celo = TransformerLM(256, args.d_model, args.nhead, args.nlayers, dim_feedforward=args.dim_feedforward).to(device)
    model_celo.load_state_dict(model_adam.state_dict())

    # --- CeLO instance ---
    celo = CeLOLite(
        hidden_sched=32,
        hidden_rule=32,
        alpha=1.0,
        lambda1=1.0,
        lambda2=1.0,
        ema_loss_beta=0.95,
        ema_grad2_beta=0.99,
        eps=1e-8,
        device=device,
    )

    # --- run CeLO ---
    print("\n=== CeLO (learned optimizer, lite) ===")
    train_dl4 = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    losses_celo, flops_celo = train_one_celo(
        model_celo, celo, train_dl4, total_steps, device, args.seq_len, args.d_model, args.dim_feedforward, args.nlayers
    )
    curves["CeLO"] = (losses_celo, flops_celo)  # <<< FIXED (was flops_velo)

    # ---------------- plots ----------------
    x = np.arange(1, total_steps + 1)
    plt.figure(figsize=(8.2, 5.0))
    for name, (losses, _) in curves.items():
        plt.plot(x[: len(losses)], losses, label=name)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss vs Steps")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(args.save_dir, f"loss_vs_steps_{run_id}.png")
    plt.savefig(out1, dpi=220)

    # Loss vs FLOPs
    plt.figure(figsize=(8.2, 5.0))
    for name, (losses, flops) in curves.items():
        n = min(len(losses), len(flops))
        plt.plot(flops[:n] / 1e9, losses[:n], label=name)
    plt.xlabel("Cumulative FLOPs (×10⁹)")
    plt.ylabel("Loss")
    plt.title("Loss vs FLOPs")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(args.save_dir, f"loss_vs_flops_{run_id}.png")
    plt.savefig(out2, dpi=220)

    # raw logs
    npz_path = os.path.join(args.save_dir, f"logs_{run_id}.npz")
    np.savez(
        npz_path,
        **{f"losses_{k}": v[0] for k, v in curves.items()},
        **{f"flops_{k}": v[1] for k, v in curves.items()},
    )

    print("Saved:\n ", out1, "\n ", out2, "\n ", npz_path)

if __name__ == "__main__":
    main()
