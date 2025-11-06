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

import matplotlib
from typing_extensions import Dict, Optional

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call
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
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(1))

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


class LSTMLOptimizerMeta(nn.Module):
    """
    Meta-trainable coordinate-wise LSTM learned optimizer that operates on
    dicts of named tensors (functional parameters) rather than nn.Parameters.
    """

    def __init__(self, feature_rms=True, hidden_size=32, eps=1e-8):
        super().__init__()
        self.feature_rms = feature_rms
        self.eps = eps
        in_dim = 2 + (1 if feature_rms else 0)  # |g|, sign(g), (optional) RMS
        self.cell = nn.LSTMCell(in_dim, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )
        self.scale_bias = nn.Parameter(torch.tensor(0.0))
        # state[name] = (h: [N,H], c: [N,H], rms: [N] or None)
        self.state = {}

    def reset_state(self):
        self.state.clear()

    def _get_state(self, name, n, H, device):
        h, c, rms = self.state.get(name, (None, None, None))
        if h is None or h.shape != (n, H):
            h = torch.zeros(n, H, device=device)
            c = torch.zeros(n, H, device=device)
        if self.feature_rms:
            if rms is None or rms.shape != (n,):
                rms = torch.zeros(n, device=device)
        else:
            rms = None
        return h, c, rms

    def _features(self, g_flat, rms_flat):
        abs_g = g_flat.abs()
        sgn_g = torch.sign(g_flat)
        if self.feature_rms:
            return torch.stack([abs_g, sgn_g, (rms_flat + 1e-12)], dim=-1)
        else:
            return torch.stack([abs_g, sgn_g], dim=-1)

    def propose_updates_from_grads(
        self,
        theta: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        lr=3e-4,
        beta_rms=0.99,
    ):
        """Return dict of updates (same shapes as theta[name]) *without* applying."""
        updates = {}
        H = self.cell.hidden_size
        for name, g in grads.items():
            if g is None:
                updates[name] = torch.zeros_like(theta[name])
                continue
            device = g.device
            g_flat = g.reshape(-1)
            n = g_flat.numel()
            h, c, rms = self._get_state(name, n, H, device)
            if self.feature_rms:
                rms = beta_rms * rms + (1.0 - beta_rms) * g_flat.pow(2)
                denom = rms.sqrt() + self.eps
            else:
                denom = g_flat.new_ones(g_flat.shape)

            feats = self._features(
                g_flat, rms if rms is not None else g_flat.new_zeros(n)
            )
            h, c = self.cell(feats, (h, c))  # (n,H), (n,H)
            s = self.head(h).squeeze(-1)  # (n,)
            scale = torch.nn.functional.softplus(s + self.scale_bias) + 1e-6
            upd_flat = -lr * scale * (g_flat / denom)  # (n,)
            updates[name] = upd_flat.view_as(theta[name])

            # keep state for next inner step
            self.state[name] = (h, c, rms if rms is not None else None)
        return updates


def _named_param_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p for n, p in module.named_parameters()}


def _clone_as_theta(named_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Make differentiable copies (leaves) for functional inner loop
    return {n: p.detach().clone().requires_grad_(True) for n, p in named_params.items()}


def meta_unroll_lstm_lo(
    model: nn.Module,
    lo_meta: LSTMLOptimizerMeta,
    data_iter,
    device,
    seq_len: int,
    unroll_steps: int = 5,
    inner_lr: float = 3e-4,
    beta_rms: float = 0.99,
    loss_mode: str = "sum",
    first_order: bool = True,
):
    """
    Do K differentiable inner steps using LO on a functional copy of model params.
    Returns (meta_loss, last_step_loss).
    """
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    base_params = _named_param_dict(model)
    theta = _clone_as_theta(base_params)  # functional weights
    lo_meta.reset_state()

    losses = []
    for k in range(unroll_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            # refresh iterator
            raise StopIteration
        x, y = x.to(device), y.to(device)
        src = x.transpose(0, 1)

        # forward under functional parameters
        logits = functional_call(model, theta, (src,), {"src_mask": mask})
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        losses.append(loss)

        # grads wrt functional params
        names = list(theta.keys())
        grads_list = torch.autograd.grad(
            loss, [theta[n] for n in names], create_graph=True, allow_unused=False
        )
        grads = {n: g for n, g in zip(names, grads_list)}

        # LO proposes updates (differentiable)
        updates = lo_meta.propose_updates_from_grads(
            theta, grads, lr=inner_lr, beta_rms=beta_rms
        )
        if first_order:
            # break the graph each inner step
            theta = {
                n: (theta[n] + updates[n]).detach().requires_grad_(True) for n in names
            }
        else:
            theta = {n: theta[n] + updates[n] for n in names}

    if loss_mode == "sum":
        meta_loss = torch.stack(losses).sum()
    elif loss_mode == "last":
        meta_loss = losses[-1]
    else:
        meta_loss = torch.stack(losses).mean()
    return meta_loss, losses[-1].detach()


def meta_train_lstm_lo(
    model_template_fn,
    lo_meta: LSTMLOptimizerMeta,
    train_dl,
    device,
    seq_len: int,
    meta_epochs: int = 1,
    meta_steps: int = 50,
    unroll_steps: int = 5,
    inner_lr: float = 3e-4,
    outer_lr: float = 1e-3,
    grad_clip: float = 1.0,
    first_order: bool = True,
):
    """
    Outer loop: sample tasks/batches, unroll inner loop, update LO weights.
    model_template_fn: lambda that returns a fresh model (same arch) on device
    """
    opt_outer = torch.optim.AdamW(lo_meta.parameters(), lr=outer_lr)
    for epoch in range(1, meta_epochs + 1):
        print(f"\n[Meta] Epoch {epoch}/{meta_epochs}")
        # fresh model per meta-step is customary; cheaper alternative is to reuse
        data_iter = iter(
            DataLoader(
                train_dl.dataset,
                batch_size=train_dl.batch_size,
                shuffle=True,
                drop_last=True,
            )
        )
        for step in range(1, meta_steps + 1):
            model = model_template_fn()  # fresh random weights
            opt_outer.zero_grad(set_to_none=True)
            try:
                meta_loss, last_loss = meta_unroll_lstm_lo(
                    model,
                    lo_meta,
                    data_iter,
                    device,
                    seq_len,
                    unroll_steps=unroll_steps,
                    inner_lr=inner_lr,
                    first_order=first_order,
                )
            except StopIteration:
                data_iter = iter(
                    DataLoader(
                        train_dl.dataset,
                        batch_size=train_dl.batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                )
                meta_loss, last_loss = meta_unroll_lstm_lo(
                    model,
                    lo_meta,
                    data_iter,
                    device,
                    seq_len,
                    unroll_steps=unroll_steps,
                    inner_lr=inner_lr,
                    first_order=first_order,
                )

            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(lo_meta.parameters(), grad_clip)
            opt_outer.step()

            if step % 5 == 0:
                print(
                    f"  [Meta] step {step:4d}/{meta_steps} | meta_loss={float(meta_loss.item()):.4f} | last_inner_loss={float(last_loss.item()):.4f}"
                )


class CeLOLite(nn.Module):
    """
    Minimal CeLO-style learned optimizer (single global scheduler + per-parameter MLP)
    with numerical safety (clamps & clipping) to prevent NaNs.
    """

    def __init__(
        self,
        hidden_sched: int = 32,
        hidden_rule: int = 32,
        alpha: float = 0.1,  # more conservative default
        lambda1: float = 1.0,
        lambda2: float = 0.1,  # more conservative default
        ema_loss_beta: float = 0.95,
        ema_grad2_beta: float = 0.99,
        eps: float = 1e-8,
        # safety knobs
        o_clip: float = 6.0,  # clamp pre-exp scheduler output
        mag_clip: float = 6.0,  # clamp pre-exp magnitude
        update_clip_ratio: float = 0.05,  # ||Δp|| <= ratio * ||p||
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
            st["v"].mul_(self.ema_grad2_beta).addcmul_(
                g, g, value=(1.0 - self.ema_grad2_beta)
            )
            gnorm_sum += float(g.norm().item())
            g2mean_sum += float(g.pow(2).mean().item())
            n_tensors += 1
        if device is None:
            return  # nothing to do

        # loss EMA (scalar)
        if not hasattr(self, "_ema_loss"):
            self._ema_loss = torch.tensor(float(loss_value), device=device)
        self._ema_loss = self._ema_loss * self.ema_loss_beta + (
            1.0 - self.ema_loss_beta
        ) * float(loss_value)

        # ---- 2) Scheduler features -> (1,4)
        prog = torch.tensor(math.log1p(step_index), device=device)
        lfeat = torch.tensor(math.log1p(float(self._ema_loss)), device=device)
        dmean = torch.tensor(gnorm_sum / max(n_tensors, 1) + 1e-12, device=device)
        rmsg = torch.tensor(math.sqrt(max(g2mean_sum, 1e-16)), device=device)
        xs = torch.stack((prog, lfeat, dmean, rmsg)).unsqueeze(0)  # (1,4)

        # ---- 3) Scheduler forward with clamp before exp
        if (
            self.sched_h is None
            or self.sched_c is None
            or self.sched_h.device != device
        ):
            H = self.sched_cell.hidden_size
            self.sched_h = torch.zeros(1, H, device=device)
            self.sched_c = torch.zeros(1, H, device=device)

        self.sched_h, self.sched_c = self.sched_cell(xs, (self.sched_h, self.sched_c))
        o_t = self.sched_head(self.sched_h).clamp(-self.o_clip, self.o_clip)  # (1,1)
        eta_t = (
            float(self.alpha) * torch.exp(o_t).squeeze()
        )  # scalar tensor in (0, alpha*e^o_clip]

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
            d_raw = self.rule_dir(h).squeeze(-1)  # (N,)
            mag = self.rule_mag(h).squeeze(-1)  # (N,)

            # safety: bound direction and magnitude
            d = torch.tanh(d_raw)  # in [-1, 1]
            mag = mag.clamp(-self.mag_clip, self.mag_clip)
            scale_mag = torch.exp(self.lambda2 * mag)  # <= exp(lambda2*mag_clip)

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


class CeLOMetaAdapter(nn.Module):
    """
    A differentiable, functional wrapper around CeLOLite. It reuses the same
    modules (sched_cell/head, rule, rule_dir/mag) but does NOT mutate model
    params in-place and does NOT keep persistent state. Instead, per-tensor
    state (m, v) and global state (sched_h/c, ema_loss) are carried by the
    caller inside a Python dict to keep the graph clean and restartable.
    """

    def __init__(self, celo_impl: CeLOLite):
        super().__init__()
        # tie to the same parameters (copy modules so .parameters() are CeLO's)
        self.sched_cell = celo_impl.sched_cell
        self.sched_head = celo_impl.sched_head
        self.rule = celo_impl.rule
        self.rule_dir = celo_impl.rule_dir
        self.rule_mag = celo_impl.rule_mag

        # hyper/safety knobs (treated as constants here)
        self.alpha = celo_impl.alpha
        self.lambda1 = celo_impl.lambda1
        self.lambda2 = celo_impl.lambda2
        self.eps = celo_impl.eps
        self.o_clip = celo_impl.o_clip
        self.mag_clip = celo_impl.mag_clip
        self.update_clip_ratio = celo_impl.update_clip_ratio
        self.ema_loss_beta = celo_impl.ema_loss_beta
        self.ema_grad2_beta = celo_impl.ema_grad2_beta
        self.rule_in = celo_impl.rule_in
        self.sched_in = celo_impl.sched_in

    def propose_updates(
        self,
        theta: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        state: Dict[str, Dict[str, torch.Tensor]],
        step_index: int,
        loss_value: torch.Tensor,
        first_order: bool = True,   # <— new
    ):
        device = next(iter(theta.values())).device
        def _maybe_detach(t):
            return t.detach() if (first_order and torch.is_tensor(t)) else t

        # shallow copy container
        next_state = {k: (v.copy() if isinstance(v, dict) else v) for k, v in state.items()}
        if "__global__" not in next_state:
            next_state["__global__"] = {}

        # ----- per-parameter EMA state (detach old state & grads if first-order) -----
        gnorm_sum = 0.0
        g2mean_sum = 0.0
        n_tensors = 0
        for name, g in grads.items():
            if g is None:
                continue
            g_use = _maybe_detach(g)
            st = next_state.get(name, None)
            if st is None:
                st = {"m": torch.zeros_like(theta[name]), "v": torch.zeros_like(theta[name])}
            m_prev = _maybe_detach(st["m"])
            v_prev = _maybe_detach(st["v"])
            m = m_prev * 0.9 + g_use * (1.0 - 0.9)
            v = v_prev * self.ema_grad2_beta + (1.0 - self.ema_grad2_beta) * (g_use * g_use)
            next_state[name] = {"m": m, "v": v}

            gnorm_sum += float(g_use.norm().item())
            g2mean_sum += float((g_use * g_use).mean().item())
            n_tensors += 1

        # ----- global scheduler state (detach hidden & ema_loss if first-order) -----
        H = self.sched_cell.hidden_size
        gh = _maybe_detach(next_state["__global__"].get("sched_h", torch.zeros(1, H, device=device)))
        gc = _maybe_detach(next_state["__global__"].get("sched_c", torch.zeros(1, H, device=device)))
        ema_loss_prev = _maybe_detach(next_state["__global__"].get(
            "ema_loss", torch.tensor(float(loss_value.item()), device=device)
        ))
        ema_loss = ema_loss_prev * self.ema_loss_beta + (1.0 - self.ema_loss_beta) * loss_value
        prog = torch.tensor(math.log1p(step_index), device=device)
        lfeat = torch.log1p(ema_loss.clamp_min(0))

        # dmean = torch.tensor(gnorm_sum / max(n_tensors, 1) + 1e-12, device=device)
        # rmsg = torch.tensor(math.sqrt(max(g2mean_sum, 1e-16)), device=device)

        sumsq, count = 0.0, 0
        for p in params:
            if p.grad is None: 
                continue
            g = p.grad
            sumsq += float(g.pow(2).sum().item())
            count += g.numel()
        dmean = torch.tensor(gnorm_sum / max(n_tensors, 1) + 1e-12, device=device)
        rmsg = torch.tensor(math.sqrt(max(sumsq / max(count, 1), 1e-16)), device=device)



        xs = torch.stack((prog, lfeat.squeeze(), dmean, rmsg)).unsqueeze(0)

        gh, gc = self.sched_cell(xs, (gh, gc))
        o_t = self.sched_head(gh).clamp(-self.o_clip, self.o_clip)
        eta_t = float(self.alpha) * torch.exp(o_t).squeeze()

        next_state["__global__"]["sched_h"] = gh
        next_state["__global__"]["sched_c"] = gc
        next_state["__global__"]["ema_loss"] = ema_loss

        # ----- compute updates (optionally detach inputs when first-order) -----
        updates: Dict[str, torch.Tensor] = {}
        for name, g in grads.items():
            if g is None:
                updates[name] = torch.zeros_like(theta[name])
                continue
            st = next_state[name]
            m, v = st["m"], st["v"]
            if first_order:
                g = g.detach(); m = m.detach(); v = v.detach()
            rms = v.sqrt() + self.eps

            feats = torch.stack([g, m, rms], dim=-1).reshape(-1, self.rule_in)
            h = self.rule(feats)
            d_raw = self.rule_dir(h).squeeze(-1)
            mag = self.rule_mag(h).squeeze(-1)

            d = torch.tanh(d_raw)
            mag = mag.clamp(-self.mag_clip, self.mag_clip)
            scale_mag = torch.exp(self.lambda2 * mag)

            upd_flat = self.lambda1 * d * scale_mag
            upd = upd_flat.view_as(theta[name])
            pn = theta[name].norm() + self.eps
            upd = eta_t * upd * pn

            u_max = self.update_clip_ratio * pn
            u_norm = upd.norm() + self.eps
            upd = torch.where(u_norm > u_max, upd * (u_max / u_norm), upd)

            updates[name] = -upd

        return updates, next_state
    # def propose_updates(
    #     self,
    #     theta: Dict[str, torch.Tensor],
    #     grads: Dict[str, torch.Tensor],
    #     state: Dict[str, Dict[str, torch.Tensor]],
    #     step_index: int,
    #     loss_value: torch.Tensor,
    #     first_order: bool = True,
    # ):
    #     """
    #     theta: dict of named functional parameters (requires_grad=True)
    #     grads: dict of grads wrt theta[name] (tensors)
    #     state: {
    #         "__global__": {"sched_h": (1,H), "sched_c": (1,H), "ema_loss": ()},
    #         <name>: {"m": tensor_like(theta[name]), "v": tensor_like(theta[name])}
    #     }
    #     Returns: (updates: Dict[name, Tensor], next_state: Dict)
    #     """
    #     device = next(iter(theta.values())).device
    #     gnorm_sum, g2mean_sum, n_tensors = 0.0, 0.0, 0
    #
    #     # ----- accumulate stats + update per-tensor EMA m,v -----
    #     next_state = {
    #         k: v.copy() if isinstance(v, dict) else v for k, v in state.items()
    #     }
    #     if "__global__" not in next_state:
    #         next_state["__global__"] = {}
    #
    #     # Per-parameter: maintain m (EMA of grad) and v (EMA of grad^2)
    #     for name, g in grads.items():
    #         if g is None:
    #             continue
    #         p_shape = theta[name].shape
    #         st = next_state.get(name, None)
    #         if st is None:
    #             st = {
    #                 "m": torch.zeros_like(theta[name]),
    #                 "v": torch.zeros_like(theta[name]),
    #             }
    #         # EMAs are differentiable; if you prefer cheaper first-order, you can .detach()
    #         m = st["m"] * 0.9 + g * (1.0 - 0.9)
    #         v = st["v"] * self.ema_grad2_beta + (1.0 - self.ema_grad2_beta) * (g * g)
    #         st = {"m": m, "v": v}
    #         next_state[name] = st
    #
    #         gnorm_sum += float(g.norm().detach().item())
    #         g2mean_sum += float((g * g).mean().detach().item())
    #         n_tensors += 1
    #
    #     # ----- global features for scheduler -----
    #     H = self.sched_cell.hidden_size
    #     gh = next_state["__global__"].get("sched_h", torch.zeros(1, H, device=device))
    #     gc = next_state["__global__"].get("sched_c", torch.zeros(1, H, device=device))
    #     ema_loss = next_state["__global__"].get(
    #         "ema_loss",
    #         torch.tensor(float(loss_value.detach().item()), device=device),
    #     )
    #     ema_loss = (
    #         ema_loss * self.ema_loss_beta + (1.0 - self.ema_loss_beta) * loss_value
    #     )
    #
    #     prog = torch.tensor(math.log1p(step_index), device=device)
    #     lfeat = torch.log1p(ema_loss.clamp_min(0))  # scalar -> stable
    #     dmean = torch.tensor(gnorm_sum / max(n_tensors, 1) + 1e-12, device=device)
    #     rmsg = torch.tensor(math.sqrt(max(g2mean_sum, 1e-16)), device=device)
    #     xs = torch.stack((prog, lfeat.squeeze(), dmean, rmsg)).unsqueeze(0)  # (1,4)
    #
    #     gh, gc = self.sched_cell(xs, (gh, gc))
    #     o_t = self.sched_head(gh).clamp(-self.o_clip, self.o_clip)  # (1,1)
    #     eta_t = float(self.alpha) * torch.exp(o_t).squeeze()  # scalar (tensor)
    #
    #     next_state["__global__"]["sched_h"] = gh
    #     next_state["__global__"]["sched_c"] = gc
    #     next_state["__global__"]["ema_loss"] = ema_loss
    #
    #     # ----- per-parameter updates (NO in-place; fully differentiable) -----
    #     updates: Dict[str, torch.Tensor] = {}
    #     for name, g in grads.items():
    #         if g is None:
    #             updates[name] = torch.zeros_like(theta[name])
    #             continue
    #         st = next_state[name]
    #         m, v = st["m"], st["v"]
    #         rms = v.sqrt() + self.eps
    #
    #         feats = torch.stack([g, m, rms], dim=-1).reshape(-1, self.rule_in)  # (N,3)
    #         h = self.rule(feats)
    #         d_raw = self.rule_dir(h).squeeze(-1)  # (N,)
    #         mag = self.rule_mag(h).squeeze(-1)
    #
    #         d = torch.tanh(d_raw)
    #         mag = mag.clamp(-self.mag_clip, self.mag_clip)
    #         scale_mag = torch.exp(self.lambda2 * mag)
    #
    #         upd_flat = self.lambda1 * d * scale_mag  # (N,)
    #         upd = upd_flat.view_as(theta[name])  # shape like param
    #
    #         pn = theta[name].norm() + self.eps
    #         upd = eta_t * upd * pn  # global schedule & scale
    #
    #         # relative clipping (uses differentiable norms)
    #         u_max = self.update_clip_ratio * pn
    #         u_norm = upd.norm() + self.eps
    #         upd = torch.where(u_norm > u_max, upd * (u_max / u_norm), upd)
    #
    #         # IMPORTANT: Do NOT apply to theta here; just return updates
    #         updates[name] = -upd
    #
    #     return updates, next_state
    #

def meta_unroll_celo(
    model: nn.Module,
    celo_meta: CeLOMetaAdapter,
    data_iter,
    device,
    seq_len: int,
    unroll_steps: int = 5,
    loss_mode: str = "sum",
    first_order: bool = True,
):
    """
    Differentiable inner loop: K steps of CeLO over a functional param dict.
    Returns (meta_loss, last_step_loss).
    """
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    base_params = {n: p for n, p in model.named_parameters()}
    theta = {n: p.detach().clone().requires_grad_(True) for n, p in base_params.items()}

    # optimizer state carried by caller; we create fresh per unroll
    state: Dict[str, Dict[str, torch.Tensor]] = {}

    losses = []
    for k in range(1, unroll_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            raise StopIteration
        x, y = x.to(device), y.to(device)
        src = x.transpose(0, 1)

        logits = functional_call(model, theta, (src,), {"src_mask": mask})
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        losses.append(loss)

        # grads wrt functional params
        names = list(theta.keys())
        grads_list = torch.autograd.grad(
            loss,
            [theta[n] for n in names],
            create_graph=not first_order,
            allow_unused=False,
        )
        grads = {n: g for n, g in zip(names, grads_list)}

        loss_for_state = loss.detach() if first_order else loss

        # CeLO proposes differentiable updates (no in-place)
        updates, state = celo_meta.propose_updates(
            theta=theta,
            grads=grads,
            state=state,
            step_index=k,
            loss_value=loss_for_state,
            first_order=first_order,
        )

        # update theta; first_order breaks graph between steps
        if first_order:
            theta = {
                n: (theta[n] + updates[n]).detach().requires_grad_(True) for n in names
            }
        else:
            theta = {n: theta[n] + updates[n] for n in names}

    if loss_mode == "sum":
        meta_loss = torch.stack(losses).sum()
    elif loss_mode == "last":
        meta_loss = losses[-1]
    else:
        meta_loss = torch.stack(losses).mean()
    return meta_loss, losses[-1].detach()


# --- add this next to meta_train_celo() ---

def _rollout_outer_objective_with_celo(
    model_template_fn,
    celo: CeLOLite,
    batch_list,              # list of (x,y) pre-fetched, identical across +/- runs
    device,
    seq_len: int,
    loss_mode: str = "sum",
):
    model = model_template_fn().to(device)
    celo.reset_state()  # critical: no state leak across rollouts

    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    losses = []
    for (x, y) in batch_list:
        # forward/backward as usual (no autograd graph needed for meta)
        for p in model.parameters():
            p.grad = None
        src = x.transpose(0, 1)
        logits = model(src, src_mask=mask)
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # CeLO in-place param update
        celo.step_inplace(list(model.parameters()), loss_value=float(loss.item()), step_index=len(losses)+1)
        losses.append(float(loss.item()))

    if loss_mode == "sum":
        return float(np.sum(losses))
    elif loss_mode == "last":
        return float(losses[-1])
    else:
        return float(np.mean(losses))


def meta_train_celo_pes(
    model_template_fn,
    celo_impl: CeLOLite,
    train_dl,
    device,
    seq_len: int,
    meta_epochs: int = 1,
    meta_steps: int = 50,
    unroll_steps: int = 5,
    sigma: float = 0.02,
    directions_per_step: int = 8,  # K directions with antithetic pairs
    outer_lr: float = 1e-3,
    loss_mode: str = "sum",
):
    """
    PES meta-training for CeLO (no autograd through inner loop).
    """
    # Outer optimizer updates CeLO params using ES gradients
    opt_outer = torch.optim.AdamW(celo_impl.parameters(), lr=outer_lr)

    # Helper to sample mirrored perturbations with the same shapes as params
    def sample_eps_like_params():
        return [torch.randn_like(p) for p in celo_impl.parameters()]

    # Pre-fetch one epoch worth of batches repeatedly as needed
    from collections import deque

    for epoch in range(1, meta_epochs + 1):
        print(f"\n[CeLO PES] Epoch {epoch}/{meta_epochs}")
        # fresh data iterator each epoch
        data_it = iter(DataLoader(train_dl.dataset,
                                  batch_size=train_dl.batch_size,
                                  shuffle=True, drop_last=True))

        for step in range(1, meta_steps + 1):
            # 1) Pre-fetch the inner unroll batches that will be reused for all rollouts this meta-step
            batch_list = []
            for i in range(unroll_steps):
                print(f"unroll step: {i}")
                try:
                    x, y = next(data_it)
                except StopIteration:
                    data_it = iter(DataLoader(train_dl.dataset,
                                              batch_size=train_dl.batch_size,
                                              shuffle=True, drop_last=True))
                    x, y = next(data_it)
                batch_list.append((x.to(device), y.to(device)))

            # 2) Accumulate ES gradient estimate
            #    We'll fill .grad on celo_impl's parameters manually.
            for p in celo_impl.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Use running buffers for grads with identical shapes
            accum_grads = [torch.zeros_like(p) for p in celo_impl.parameters()]

            for i in range(directions_per_step):
                print(f"direction: {i}")
                eps = sample_eps_like_params()

                with torch.no_grad():
                    # + direction
                    for p, e in zip(celo_impl.parameters(), eps):
                        p.add_(sigma * e)
                J_plus = _rollout_outer_objective_with_celo(
                    model_template_fn, celo_impl, batch_list, device, seq_len, loss_mode
                )

                with torch.no_grad():
                    # - direction (mirror)
                    for p, e in zip(celo_impl.parameters(), eps):
                        p.add_(-2.0 * sigma * e)  # from (θ+σε) -> (θ-σε)
                J_minus = _rollout_outer_objective_with_celo(
                    model_template_fn, celo_impl, batch_list, device, seq_len, loss_mode
                )

                with torch.no_grad():
                    # restore to original θ
                    for p, e in zip(celo_impl.parameters(), eps):
                        p.add_(sigma * e)

                # ES gradient contribution: ((J+ - J-) / (2σ)) * ε
                coeff = (J_plus - J_minus) / (2.0 * sigma)
                for gbuf, e in zip(accum_grads, eps):
                    gbuf.add_(coeff * e)

            # Average over directions and assign as gradients
            scale = 1.0 / float(directions_per_step)
            for p, gbuf in zip(celo_impl.parameters(), accum_grads):
                p.grad = (scale * gbuf)

            # Optional: gradient clipping on meta-grad (usually not needed)
            torch.nn.utils.clip_grad_norm_(celo_impl.parameters(), max_norm=1.0)

            opt_outer.step()
            opt_outer.zero_grad(set_to_none=True)

            if (step % 5) == 0:
                print(f"  [CeLO PES] step {step:4d}/{meta_steps} | J+= {J_plus:.4f} | J-= {J_minus:.4f}")

def meta_train_celo(
    model_template_fn,
    celo_impl: CeLOLite,
    train_dl,
    device,
    seq_len: int,
    meta_epochs: int = 1,
    meta_steps: int = 50,
    unroll_steps: int = 5,
    outer_lr: float = 1e-3,
    grad_clip: float = 1.0,
    first_order: bool = True,
):
    """
    Outer loop that updates CeLO's weights using differentiable unrolls.
    """
    celo_meta = CeLOMetaAdapter(celo_impl).to(device)
    opt_outer = torch.optim.AdamW(celo_meta.parameters(), lr=outer_lr)

    for epoch in range(1, meta_epochs + 1):
        print(f"\n[CeLO Meta] Epoch {epoch}/{meta_epochs}")
        data_iter = iter(
            DataLoader(
                train_dl.dataset,
                batch_size=train_dl.batch_size,
                shuffle=True,
                drop_last=True,
            )
        )
        for step in range(1, meta_steps + 1):
            model = model_template_fn().to(device)
            opt_outer.zero_grad(set_to_none=True)

            try:
                meta_loss, last_loss = meta_unroll_celo(
                    model,
                    celo_meta,
                    data_iter,
                    device,
                    seq_len,
                    unroll_steps=unroll_steps,
                    first_order=first_order,
                )
            except StopIteration:
                data_iter = iter(
                    DataLoader(
                        train_dl.dataset,
                        batch_size=train_dl.batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                )
                meta_loss, last_loss = meta_unroll_celo(
                    model,
                    celo_meta,
                    data_iter,
                    device,
                    seq_len,
                    unroll_steps=unroll_steps,
                    first_order=first_order,
                )

            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(celo_meta.parameters(), grad_clip)
            opt_outer.step()

            if step % 5 == 0:
                print(
                    f"  [CeLO Meta] step {step:4d}/{meta_steps} | meta_loss={float(meta_loss.item()):.4f} | last_inner_loss={float(last_loss.item()):.4f}"
                )


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
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )
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
        return (
            torch.stack([abs_g, sgn_g, rms_flat.clamp_min(1e-12)], dim=-1)
            if self.feature_rms
            else torch.stack([abs_g, sgn_g], dim=-1)
        )

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
            feats = self._features_from_grad(
                g_flat, rms if rms is not None else g_flat.new_zeros(g_flat.shape)
            )
            h, c = self.cell(feats, (h, c))
            s = self.head(h).squeeze(-1)
            scale = torch.nn.functional.softplus(s + self.scale_bias) + 1e-6
            upd_flat = -lr * scale * (g_flat / denom)
            updates.append(upd_flat.view_as(p))
            self._set_state_tensors(
                p, h.detach(), c.detach(), rms.detach() if rms is not None else None
            )
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
    def __init__(
        self,
        params,
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        gamma=1e-3,
        eps=1e-8,
        eps2=1e-12,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            gamma=gamma,
            eps=eps,
            eps2=eps2,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta1, beta2, gamma, eps, eps2, wd = (
                group["lr"],
                group["beta1"],
                group["beta2"],
                group["gamma"],
                group["eps"],
                group["eps2"],
                group["weight_decay"],
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


def train_one_celo(
    model,
    celo: CeLOLite,
    train_dl,
    steps,
    device,
    seq_len,
    d_model,
    dim_ff,
    nlayers,
    log_every=10,
):
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
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # CeLO update (in-place)
        celo.step_inplace(
            list(model.parameters()), loss_value=float(loss.item()), step_index=t
        )

        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")

    return np.array(losses, float), np.array(flops, float)


def train_one_torchopt(
    model,
    optimizer,
    train_dl,
    steps,
    device,
    seq_len,
    d_model,
    dim_ff,
    nlayers,
    log_every=10,
):
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
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")
    return np.array(losses, float), np.array(flops, float)


def train_one_lstm(
    model,
    learned_opt: LSTMLOptimizer,
    base_lr,
    train_dl,
    steps,
    device,
    seq_len,
    d_model,
    dim_ff,
    nlayers,
    log_every=10,
):
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
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learned_opt.step_inplace(list(model.parameters()), lr=base_lr, beta_rms=0.99)
        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")
    return np.array(losses, float), np.array(flops, float)


def train_one_pylo(
    model,
    optimizer,
    train_dl,
    steps,
    device,
    seq_len,
    d_model,
    dim_ff,
    nlayers,
    log_every=10,
):
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
        loss = criterion(
            logits.transpose(0, 1).reshape(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(loss)  # <-- PyLO requires the loss
        cum_flops += C_step
        losses.append(float(loss.item()))
        flops.append(cum_flops)
        if (t % log_every) == 0:
            print(f" step {t:5d} | loss {loss.item():.4f}")
    return np.array(losses, float), np.array(flops, float)


def _parse_models_arg(raw: str):
    """
    Accepts comma/space-separated names or 'all'.
    Valid: adamw, adasign, lstm_lo, celo, velo, adafaclo, mulo
    """
    if raw.strip().lower() == "all":
        return ["adamw", "adasign", "lstm_lo", "celo", "velo", "adafaclo"]
    # split on comma or spaces
    parts = [p.strip().lower() for chunk in raw.split(",") for p in chunk.split()]
    valid = {"adamw", "adasign", "lstm_lo", "celo", "velo", "adafaclo", "mulo"}
    out = []
    for p in parts:
        if p not in valid:
            raise argparse.ArgumentTypeError(
                f"Unknown model '{p}'. Choose from {sorted(valid)} or 'all'."
            )
        out.append(p)
    # de-dup but preserve order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def build_argparser():
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

    ap.add_argument(
        "--celo_init", choices=["random", "heuristic", "load"], default="random"
    )
    ap.add_argument("--celo_ckpt", type=str, default="", help="CeLO state_dict to load")
    ap.add_argument(
        "--celo_save_ckpt",
        type=str,
        default="",
        help="Where to save CeLO state_dict after run",
    )

    # --- Which models to run (new) ---
    ap.add_argument(
        "--models",
        type=_parse_models_arg,
        default=_parse_models_arg("adamw adasign lstm_lo celo velo adafaclo"),
        help="Comma/space list or 'all'. Options: adamw, adasign, lstm_lo, celo, velo, adafaclo, mulo",
    )

    # --- AdaSign-Lite hypers (unchanged) ---
    ap.add_argument("--asl_beta1", type=float, default=0.9)
    ap.add_argument("--asl_beta2", type=float, default=0.999)
    ap.add_argument("--asl_gamma", type=float, default=1e-3)
    ap.add_argument("--asl_eps", type=float, default=1e-8)
    ap.add_argument("--asl_eps2", type=float, default=1e-12)

    # --- LSTM-LO hypers (existing) ---
    ap.add_argument("--lstm_hidden", type=int, default=32)
    ap.add_argument("--lstm_feature_rms", action="store_true", default=True)

    # --- NEW: LSTM-LO initialization mode ---
    ap.add_argument(
        "--lstm_lo_init",
        choices=["random", "meta", "load"],
        default="random",
        help="How to initialize the LSTM learned optimizer used in training.",
    )
    ap.add_argument(
        "--lstm_lo_ckpt",
        type=str,
        default="",
        help="Path to .pt for --lstm_lo_init=load (state_dict of LSTMLOptimizer or LSTMLOptimizerMeta).",
    )
    ap.add_argument(
        "--lstm_lo_save_ckpt",
        type=str,
        default="",
        help="If set and --lstm_lo_init=meta, save the meta-trained LO state_dict here.",
    )

    # ---- Meta-training controls for LSTM-LO (unchanged) ----
    ap.add_argument("--lstm_meta_epochs", type=int, default=1)
    ap.add_argument("--lstm_meta_steps", type=int, default=50)
    ap.add_argument("--lstm_meta_unroll", type=int, default=5)
    ap.add_argument("--lstm_meta_inner_lr", type=float, default=3e-4)
    ap.add_argument("--lstm_meta_outer_lr", type=float, default=1e-3)
    ap.add_argument("--lstm_meta_first_order", action="store_true", default=True)

    # in build_argparser()
    ap.add_argument(
        "--meta_train_celo",
        action="store_true",
        default=False,
        help="Meta-train CeLO before evaluation and use the trained weights.",
    )
    ap.add_argument("--celo_meta_epochs", type=int, default=1)
    ap.add_argument("--celo_meta_steps", type=int, default=50)
    ap.add_argument("--celo_meta_unroll", type=int, default=5)
    ap.add_argument("--celo_meta_outer_lr", type=float, default=1e-3)
    ap.add_argument("--celo_meta_first_order", action="store_true", default=True)

    return ap


# --------------------------- main ---------------------------


def main():
    ap = build_argparser()
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
    print("Models to run:", args.models)

    # Data
    train_ds = ByteLMDataset(args.data_path, seq_len=args.seq_len, step=1)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Base model factory
    def make_model():
        return TransformerLM(
            256,
            args.d_model,
            args.nhead,
            args.nlayers,
            dim_feedforward=args.dim_feedforward,
        ).to(device)

    # ----------------- LSTM-LO init selection -----------------
    learned_opt = None
    if "lstm_lo" in args.models:
        if args.lstm_lo_init == "meta":
            print("\n=== Meta-training LSTM Learned Optimizer ===")
            lo_meta = LSTMLOptimizerMeta(
                feature_rms=args.lstm_feature_rms, hidden_size=args.lstm_hidden
            ).to(device)
            meta_train_lstm_lo(
                make_model,
                lo_meta,
                train_dl,
                device,
                args.seq_len,
                meta_epochs=args.lstm_meta_epochs,
                meta_steps=args.lstm_meta_steps,
                unroll_steps=args.lstm_meta_unroll,
                inner_lr=args.lstm_meta_inner_lr,
                outer_lr=args.lstm_meta_outer_lr,
                first_order=args.lstm_meta_first_order,
            )
            if args.lstm_lo_save_ckpt:
                torch.save(lo_meta.state_dict(), args.lstm_lo_save_ckpt)
                print(f"[Meta] Saved meta-trained LSTM-LO to: {args.lstm_lo_save_ckpt}")

            learned_opt = LSTMLOptimizer(
                feature_rms=args.lstm_feature_rms, hidden_size=args.lstm_hidden
            ).to(device)
            learned_opt.load_state_dict(lo_meta.state_dict())

        elif args.lstm_lo_init == "load":
            if not args.lstm_lo_ckpt or not os.path.exists(args.lstm_lo_ckpt):
                raise FileNotFoundError(
                    "Specify a valid --lstm_lo_ckpt when --lstm_lo_init=load"
                )
            print(f"\n=== Loading LSTM-LO state from {args.lstm_lo_ckpt} ===")
            sd = torch.load(args.lstm_lo_ckpt, map_location=device)
            learned_opt = LSTMLOptimizer(
                feature_rms=args.lstm_feature_rms, hidden_size=args.lstm_hidden
            ).to(device)
            learned_opt.load_state_dict(sd)

        else:  # random
            print("\n=== Using randomly initialized LSTM-LO ===")
            learned_opt = LSTMLOptimizer(
                feature_rms=args.lstm_feature_rms, hidden_size=args.lstm_hidden
            ).to(device)

    total_steps = args.epochs * args.steps_per_epoch
    print(f"Training {total_steps} steps per selected model...")

    curves = {}

    # -------- AdamW --------
    if "adamw" in args.models:
        print("\n=== AdamW ===")
        model_adam = make_model()
        opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=args.lr)
        losses_adam, flops_adam = train_one_torchopt(
            model_adam,
            opt_adam,
            train_dl,
            total_steps,
            device,
            args.seq_len,
            args.d_model,
            args.dim_feedforward,
            args.nlayers,
        )
        curves["AdamW"] = (losses_adam, flops_adam)
    else:
        model_adam = None  # may be used as a weight seed below

    # -------- AdaSign-Lite --------
    if "adasign" in args.models:
        print("\n=== AdaSign-Lite ===")
        model_asl = make_model()
        if model_adam is not None:
            model_asl.load_state_dict(model_adam.state_dict())
        opt_asl = AdaSignLite(
            model_asl.parameters(),
            lr=args.lr,
            beta1=args.asl_beta1,
            beta2=args.asl_beta2,
            gamma=args.asl_gamma,
            eps=args.asl_eps,
            eps2=args.asl_eps2,
        )
        train_dl2 = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        losses_asl, flops_asl = train_one_torchopt(
            model_asl,
            opt_asl,
            train_dl2,
            total_steps,
            device,
            args.seq_len,
            args.d_model,
            args.dim_feedforward,
            args.nlayers,
        )
        curves["AdaSign-Lite"] = (losses_asl, flops_asl)

    # -------- LSTM-LO (learned optimizer) --------
    if "lstm_lo" in args.models:
        print("\n=== LSTM Learned Optimizer ===")
        model_lstm = make_model()
        if model_adam is not None:
            model_lstm.load_state_dict(model_adam.state_dict())
        train_dl3 = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        losses_lstm, flops_lstm = train_one_lstm(
            model_lstm,
            learned_opt,
            args.lr,
            train_dl3,
            total_steps,
            device,
            args.seq_len,
            args.d_model,
            args.dim_feedforward,
            args.nlayers,
        )
        curves["LSTM-LO"] = (losses_lstm, flops_lstm)

    # -------- CeLO (lite) --------
    if "celo" in args.models:
        print("\n=== CeLO (learned optimizer, lite) ===")
        model_celo = make_model()
        if model_adam is not None:
            model_celo.load_state_dict(model_adam.state_dict())

        # single construction — do NOT rebuild later
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

        # --- optional helpers ---
        def init_celo_as_rms_sgd(celo: CeLOLite, base_lr=3e-4):
            with torch.no_grad():
                # scheduler ~ constant step size: exp(o_t) ~ base_lr/alpha
                for p in celo.sched_cell.parameters():
                    p.zero_()
                celo.sched_head.weight.zero_()
                celo.sched_head.bias.fill_(math.log(max(base_lr / celo.alpha, 1e-6)))

                # rule: direction ≈ -sign(g), magnitude small
                for m in celo.rule:
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                nn.init.zeros_(celo.rule_dir.weight)
                celo.rule_dir.bias.fill_(-1.0)  # tanh(-1) ~ -0.76
                nn.init.zeros_(celo.rule_mag.weight)
                celo.rule_mag.bias.fill_(-2.0)  # exp(λ2*mag) small

        # --- initialization path ---
        if args.celo_init == "load":
            if not args.celo_ckpt or not os.path.exists(args.celo_ckpt):
                raise FileNotFoundError("Provide --celo_ckpt for --celo_init=load")
            sd = torch.load(args.celo_ckpt, map_location=device)
            celo.load_state_dict(sd)
            print(f"[CeLO] Loaded weights from {args.celo_ckpt}")
        elif args.celo_init == "heuristic":
            init_celo_as_rms_sgd(celo, base_lr=args.lr)
            print("[CeLO] Initialized to RMS-normalized SGD-like heuristic")
        else:
            print("[CeLO] Using random weights (no meta-training / loading)")

        if args.meta_train_celo:
            print("\n=== Meta-training CeLO ===")
            meta_train_celo_pes(
                model_template_fn=make_model,
                celo_impl=celo,
                train_dl=train_dl,
                device=device,
                seq_len=args.seq_len,
                meta_epochs=args.celo_meta_epochs,
                meta_steps=args.celo_meta_steps,
                unroll_steps=args.celo_meta_unroll,
                sigma=0.02,
                directions_per_step=8,
                outer_lr=args.celo_meta_outer_lr,
                loss_mode="sum",   # or "last"/"mean" to match your plots
            )

        if args.celo_save_ckpt:
            torch.save(celo.state_dict(), args.celo_save_ckpt)
            print(f"[CeLO Meta] Saved meta-trained CeLO to: {args.celo_save_ckpt}")

        # --- train ---
        train_dl4 = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        losses_celo, flops_celo = train_one_celo(
            model_celo,
            celo,
            train_dl4,
            total_steps,
            device,
            args.seq_len,
            args.d_model,
            args.dim_feedforward,
            args.nlayers,
        )
        curves["CeLO"] = (losses_celo, flops_celo)

        # optional: save the CeLO weights you used (handy for reproducing)
        if args.celo_save_ckpt:
            torch.save(celo.state_dict(), args.celo_save_ckpt)
            print(f"[CeLO] Saved weights to {args.celo_save_ckpt}")

    # -------- PyLO family --------
    if "velo" in args.models or "adafaclo" in args.models or "mulo" in args.models:
        if not PYLO_AVAILABLE:
            print(">> PyLO not installed; skipping all PyLO models.")
        else:
            # VeLO
            if "velo" in args.models:
                print("\n=== PyLO VeLO ===")
                model_velo = make_model()
                if model_adam is not None:
                    model_velo.load_state_dict(model_adam.state_dict())
                opt_velo = VeLO(model_velo.parameters())
                train_dlV = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
                )
                losses_velo, flops_velo = train_one_pylo(
                    model_velo,
                    opt_velo,
                    train_dlV,
                    total_steps,
                    device,
                    args.seq_len,
                    args.d_model,
                    args.dim_feedforward,
                    args.nlayers,
                )
                curves["VeLO"] = (losses_velo, flops_velo)

            # AdafacLO
            if "adafaclo" in args.models:
                print("\n=== PyLO AdafacLO (naive) ===")
                model_af = make_model()
                if model_adam is not None:
                    model_af.load_state_dict(model_adam.state_dict())
                opt_af = AdafacLO_naive(model_af.parameters())
                train_dlA = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
                )
                losses_af, flops_af = train_one_pylo(
                    model_af,
                    opt_af,
                    train_dlA,
                    total_steps,
                    device,
                    args.seq_len,
                    args.d_model,
                    args.dim_feedforward,
                    args.nlayers,
                )
                curves["AdafacLO"] = (losses_af, flops_af)

            # MuLO (may require μP-consistent shapes)
            if "mulo" in args.models:
                print("\n=== PyLO MuLO (μP wrapper over AdafacLO) ===")
                try:
                    model_mulo = make_model()
                    if model_adam is not None:
                        model_mulo.load_state_dict(model_adam.state_dict())
                    opt_mulo = MuLO_naive(model_mulo.parameters())
                    train_dlM = DataLoader(
                        train_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                    losses_mulo, flops_mulo = train_one_pylo(
                        model_mulo,
                        opt_mulo,
                        train_dlM,
                        total_steps,
                        device,
                        args.seq_len,
                        args.d_model,
                        args.dim_feedforward,
                        args.nlayers,
                    )
                    curves["MuLO"] = (losses_mulo, flops_mulo)
                except Exception as e:
                    print(f">> MuLO run skipped (needs μP shapes): {e}")

    # ---------------- plots/logs ----------------
    x = np.arange(1, total_steps + 1)
    if len(curves) == 0:
        print("No models were run. Nothing to plot/log.")
        return

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

    npz_path = os.path.join(args.save_dir, f"logs_{run_id}.npz")
    np.savez(
        npz_path,
        **{f"losses_{k}": v[0] for k, v in curves.items()},
        **{f"flops_{k}": v[1] for k, v in curves.items()},
    )

    print("Saved:\n ", out1, "\n ", out2, "\n ", npz_path)


if __name__ == "__main__":
    main()
