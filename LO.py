#!/usr/bin/env python3
"""
AdamW vs AdaSign-Lite vs LSTM Learned Optimizer on enwik8 (byte-level).

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pylo.optim import VeLO

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


# --------------------------- FLOPs proxy ---------------------------


def fwd_flops(B, S, D, F, L):
    # rough transformer encoder forward FLOPs
    return L * (4 * B * S * S * D + 8 * B * S * D * D + 4 * B * S * D * F)


# --------------------------- LSTM Learned Optimizer ---------------------------


class LSTMLOptimizer(nn.Module):
    """
    Coordinate-wise LSTM learned optimizer (shared weights).
    Input per coordinate: [|g|, sign(g), rms] -> LSTM -> head -> scale >= 0
    Update: - lr * scale * g / (sqrt(rms) + eps)
    """

    def __init__(self, feature_rms=True, hidden_size=32, eps=1e-8):
        super().__init__()
        self.feature_rms = feature_rms
        in_dim = 2 + (1 if feature_rms else 0)
        self.cell = nn.LSTMCell(in_dim, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.scale_bias = nn.Parameter(torch.tensor(0.0))
        self.eps = eps
        self._state = {}  # p -> (h, c, rms)

    def reset_state(self):
        self._state.clear()

    @torch.no_grad()
    def _get_state_tensors(self, p, hidden_size, device):
        h, c, rms = self._state.get(p, (None, None, None))
        n = p.numel()
        if h is None or h.numel() != n * hidden_size:
            h = torch.zeros(n, hidden_size, device=device)
            c = torch.zeros(n, hidden_size, device=device)
        if self.feature_rms:
            if rms is None or rms.numel() != n:
                rms = torch.zeros(n, device=device)
        else:
            rms = None
        return h, c, rms

    def _set_state_tensors(self, p, h, c, rms):
        self._state[p] = (h, c, rms)

    def _features_from_grad(self, g_flat, rms_flat):
        abs_g = g_flat.abs()
        sgn_g = torch.sign(g_flat)
        if self.feature_rms:
            return torch.stack([abs_g, sgn_g, rms_flat.clamp_min(1e-12)], dim=-1)
        else:
            return torch.stack([abs_g, sgn_g], dim=-1)

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


# --------------------------- AdaSign-Lite Optimizer ---------------------------


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

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                m, v, s = state["m"], state["v"], state["s"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                denom = v.sqrt() + eps
                align = (g * m) / (denom + eps2)
                s.add_(align, alpha=gamma)

                step_scale = torch.exp(torch.clamp(s, -10, 10))
                update = (lr * step_scale) * (g / (denom + eps2))
                p.add_(-update)
        return None


# --------------------------- training helpers ---------------------------


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
    """For torch.optim-style optimizers (AdamW, AdaSign-Lite)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    B, S, D, F, L = train_dl.batch_size, seq_len, d_model, dim_ff, nlayers
    C_fwd = fwd_flops(B, S, D, F, L)
    C_bwd = 2.0 * C_fwd
    C_step = C_fwd + C_bwd
    cum_flops = 0.0

    losses, flops = [], []
    it = iter(train_dl)

    for t in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
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
    """For the functional LSTM learned optimizer."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    B, S, D, F, L = train_dl.batch_size, seq_len, d_model, dim_ff, nlayers
    C_fwd = fwd_flops(B, S, D, F, L)
    C_bwd = 2.0 * C_fwd
    C_step = C_fwd + C_bwd
    cum_flops = 0.0

    losses, flops = [], []
    it = iter(train_dl)
    learned_opt.reset_state()

    for t in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
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


# --------------------------- main ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="enwik8")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=500)
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

    # Data
    train_ds = ByteLMDataset(args.data_path, seq_len=args.seq_len, step=1)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Models (identical init)
    model_adam = TransformerLM(
        256,
        args.d_model,
        args.nhead,
        args.nlayers,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    model_asl = TransformerLM(
        256,
        args.d_model,
        args.nhead,
        args.nlayers,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    model_lstm = TransformerLM(
        256,
        args.d_model,
        args.nhead,
        args.nlayers,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    model_asl.load_state_dict(model_adam.state_dict())
    model_lstm.load_state_dict(model_adam.state_dict())

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
    learned_opt = LSTMLOptimizer(
        feature_rms=args.lstm_feature_rms, hidden_size=args.lstm_hidden
    ).to(device)

    total_steps = args.epochs * args.steps_per_epoch
    print(f"Training {total_steps} steps per run...")

    # AdamW
    print("\n=== AdamW ===")
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

    # AdaSign-Lite
    print("\n=== AdaSign-Lite (learnable) ===")
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

    # LSTM Learned Optimizer
    print("\n=== LSTM Learned Optimizer ===")
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

    # ---------------- plots ----------------
    # Loss vs steps
    plt.figure(figsize=(7.6, 4.8))
    x = np.arange(1, total_steps + 1)
    plt.plot(x, losses_adam, label="AdamW")
    plt.plot(x, losses_asl, label="AdaSign-Lite")
    plt.plot(x, losses_lstm, label="LSTM-LO")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Loss vs Steps")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(args.save_dir, f"loss_vs_steps_{run_id}.png")
    plt.savefig(out1, dpi=220)

    # Loss vs FLOPs
    n = min(
        len(flops_adam),
        len(losses_adam),
        len(flops_asl),
        len(losses_asl),
        len(flops_lstm),
        len(losses_lstm),
    )
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(flops_adam[:n] / 1e9, losses_adam[:n], label="AdamW")
    plt.plot(flops_asl[:n] / 1e9, losses_asl[:n], label="AdaSign-Lite")
    plt.plot(flops_lstm[:n] / 1e9, losses_lstm[:n], label="LSTM-LO")
    plt.xlabel("Cumulative FLOPs (×10⁹)")
    plt.ylabel("Loss")
    plt.title("Loss vs FLOPs")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(args.save_dir, f"loss_vs_flops_{run_id}.png")
    plt.savefig(out2, dpi=220)

    # raw logs
    np.savez(
        os.path.join(args.save_dir, f"logs_{run_id}.npz"),
        losses_adam=losses_adam,
        flops_adam=flops_adam,
        losses_asl=losses_asl,
        flops_asl=flops_asl,
        losses_lstm=losses_lstm,
        flops_lstm=flops_lstm,
    )

    print("Saved:\n ", out1, "\n ", out2)


if __name__ == "__main__":
    main()
