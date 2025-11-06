#!/usr/bin/env python3
import argparse, os, math, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- import your existing building blocks from LO.py ----
# (CeLOLite + PES meta-training + tiny transformer + enwik8 loader)
from LO import (
    CeLOLite,
    meta_train_celo_pes,
    TransformerLM,
    ByteLMDataset,
    download_enwik8,
)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="enwik8", help="raw enwik8 file")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # PES meta-training knobs (fast & stable for CeLO)
    ap.add_argument("--meta_epochs", type=int, default=1)
    ap.add_argument("--meta_steps", type=int, default=50)
    ap.add_argument("--unroll_steps", type=int, default=5)
    ap.add_argument("--outer_lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--directions", type=int, default=8)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save_ckpt", type=str, default="celo_base.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Download + build a training dataset (use most of enwik8 for meta-train batches)
    download_enwik8(args.data_path)
    train_ds = ByteLMDataset(args.data_path, seq_len=args.seq_len, step=args.seq_len)  # non-overlap windows
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    def make_model():
        return TransformerLM(
            vocab_size=256,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dim_feedforward=args.dim_ff,
        ).to(device)

    # Initialize CeLO (lite) and meta-train with PES (the function is already in LO.py)
    celo = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=0.1, lambda1=1.0, lambda2=0.1, device=device)

    meta_train_celo_pes(
        model_template_fn=make_model,
        celo_impl=celo,
        train_dl=train_dl,
        device=device,
        seq_len=args.seq_len,
        meta_epochs=args.meta_epochs,
        meta_steps=args.meta_steps,
        unroll_steps=args.unroll_steps,
        sigma=args.sigma,
        directions_per_step=args.directions,
        outer_lr=args.outer_lr,
        loss_mode="sum",
    )

    os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
    torch.save(celo.state_dict(), args.save_ckpt)
    print(f"[DONE] Saved CeLO checkpoint to: {args.save_ckpt}")

if __name__ == "__main__":
    main()
