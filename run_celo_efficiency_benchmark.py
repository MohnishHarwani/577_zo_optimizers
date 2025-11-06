#!/usr/bin/env python3
import argparse, os, time, math, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

import numpy as np
import jax
from huggingface_hub import hf_hub_download
from celo.factory import get_optimizer
from celo.utils import init_lopt_from_ckpt

from collections import deque
# --- optional PyLO / VeLO ---
PYLO_AVAILABLE = True
from pylo.optim import VeLO  # pip/conda install pylo (your PyLO fork/build)

def time_to_target_celo_hf(model, celo_adapter, train_dl, val_dl, seq_len, device,
                           target_val_loss, max_steps, eval_every, args):
    model.train()
    crit = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    start = time.perf_counter()
    steps = 0
    it = iter(train_dl)
    ema = Smoothed(alpha=0.98)
    last_val = float("inf")
    tokens_per_step = seq_len * train_dl.batch_size

    while steps < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0,1)

        for p in model.parameters(): p.grad = None
        logits = model(src, src_mask=mask)
        loss = crit(logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
        ema_loss = ema.update(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ---- JAX CeLO pretrained step (in-place param update) ----
        celo_adapter.step_inplace(model, loss_value=float(loss.item()), step_index=steps+1)

        steps += 1

        if (steps % args.log_every) == 0:
            # (optional throughput/memory print, matching your style)
            print(f"[CeLO-HF] step {steps:5d} | train_loss {loss.item():.4f} | ema {ema_loss:.4f} "
                  f"| val {last_val:.4f} | elapsed {fmt_secs(time.perf_counter()-start)}", flush=True)

        if (steps % eval_every) == 0:
            v = eval_val_loss(model, val_dl, seq_len, device)
            last_val = v
            print(f"[CeLO-HF][eval] step {steps:5d} | val_loss {v:.4f} | target {target_val_loss:.4f}",
                  flush=True)
            if v <= target_val_loss:
                print(f"[CeLO-HF][done] step {steps} | time {fmt_secs(time.perf_counter()-start)} | val {v:.4f}",
                      flush=True)
                return time.perf_counter() - start, steps, v

    return None, steps, eval_val_loss(model, val_dl, seq_len, device)

class CeloJaxAdapter:
    """
    Wraps pretrained CeLO (JAX) to update a PyTorch model.
    On each step:
      - read PyTorch grads
      - call CeLO.update in JAX
      - compute parameter deltas and apply to Torch model in-place
    """
    def __init__(self, model, repo="amoudgl/celo", filename="theta.state", num_steps=1000):

        # Snapshot initial params (numpy on CPU) keyed by Torch names
        self._names = []
        self._params_np = {}
        for n, p in model.named_parameters():
            arr = np.array(p.detach().cpu().numpy(), copy=True)
            self._params_np[n] = arr
            self._names.append(n)

        # Download checkpoint & construct optimizer
        ckpt_path = hf_hub_download(repo_id=repo, filename=filename, local_dir="./")
        lopt = get_optimizer('celo')
        self.opt = init_lopt_from_ckpt(lopt, ckpt_path)

        # Initialize CeLO state (no model_state for generic PyTorch models)
        self.state = self.opt.init(params=self._params_np, model_state=None, num_steps=num_steps)

    def step_inplace(self, model, loss_value: float, step_index: int):
        # Build grad dict from current Torch grads
        grads_np = {}
        for n, p in model.named_parameters():
            g = p.grad
            if g is None:
                # If a grad is missing, treat as zeros of the same shape
                grads_np[n] = np.zeros_like(self._params_np[n])
            else:
                grads_np[n] = np.array(g.detach().cpu().numpy(), copy=False)

        # JAX CeLO update
        self.state = self.opt.update(self.state, grad=grads_np, loss=loss_value)

        # Get new param snapshot from CeLO and apply delta to Torch
        new_params = self.opt.get_params(self.state)
        with torch.no_grad():
            for n, p in model.named_parameters():
                old_np = self._params_np[n]
                new_np = new_params[n]
                delta = torch.from_numpy(new_np - old_np).to(p.device, dtype=p.dtype)
                p.add_(delta)
                # keep mirror in sync
                self._params_np[n] = new_np


def fmt_secs(s: float) -> str:
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class Smoothed:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.val = None
    def update(self, x: float):
        self.val = x if self.val is None else self.alpha*self.val + (1-self.alpha)*x
        return self.val

# Reuse your code
from LO import (
    CeLOLite, CeLOMetaAdapter,
    TransformerLM, ByteLMDataset, download_enwik8,
    generate_square_subsequent_mask
)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_model(d_model, nhead, nlayers, dim_ff, device):
    return TransformerLM(256, d_model, nhead, nlayers, dim_ff).to(device)

@torch.no_grad()
def eval_val_loss(model, val_dl, seq_len, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="mean")
    mask = generate_square_subsequent_mask(seq_len).to(device)
    losses = []
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        src = x.transpose(0,1)
        logits = model(src, src_mask=mask)
        loss = crit(logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")

def build_data(data_path, seq_len, batch_size):
    download_enwik8(data_path)
    full = ByteLMDataset(data_path, seq_len=seq_len, step=seq_len)
    # simple split: 90% train, 10% val
    N = len(full)
    N_train = max(1, int(0.9 * N))
    train_idx = range(0, N_train)
    val_idx   = range(N_train, N)
    # deterministic, no shuffle for fairness (same order for every run)
    train_dl = DataLoader(torch.utils.data.Subset(full, list(train_idx)), batch_size=batch_size, shuffle=False, drop_last=True)
    val_dl   = DataLoader(torch.utils.data.Subset(full, list(val_idx)),   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl

# -------------------------------
# CeLO compression utilities
# -------------------------------
def apply_quantization_fp16(celo: CeLOLite):
    # Convert all learnable modules to fp16; use autocast in training loop.
    celo.half()
    return celo, dict(use_amp=True)

def apply_dynamic_int8_linear_only(celo: CeLOLite):
    # Safe dynamic quantization on Linear layers (LSTMCell dynamic quant not supported)
    try:
        from torch.ao.quantization import quantize_dynamic
        import torch.nn as nn
        modules = {nn.Linear}
        q_celo = quantize_dynamic(celo, {nn.Linear}, dtype=torch.qint8)
        return q_celo, dict(use_amp=False)  # int8 path runs on CPU; not ideal on GPU
    except Exception as e:
        warnings.warn(f"Dynamic int8 quantization unavailable: {e}")
        return celo, dict(use_amp=False)

def apply_pruning_l1(celo: CeLOLite, amount: float):
    # Prune L1 on key weights; keep biases intact. Mask is persistent via prune module.
    # Scheduler head + rule MLP + dir/mag, and LSTMCell ih/hh
    targets = []
    # LSTMCell
    targets += [(celo.sched_cell, "weight_ih"), (celo.sched_cell, "weight_hh")]
    # Rule MLP (Sequential of Linear/Tanh)
    for m in celo.rule:
        if isinstance(m, nn.Linear):
            targets += [(m, "weight")]
    # Heads
    targets += [(celo.rule_dir, "weight"), (celo.rule_mag, "weight"), (celo.sched_head, "weight")]
    for mod, name in targets:
        try:
            prune.l1_unstructured(mod, name=name, amount=amount)
        except Exception as e:
            warnings.warn(f"Prune failed for {mod}.{name}: {e}")
    return celo

# -------------------------------
# Training loops (time-to-target)
# -------------------------------
def time_to_target_celo(model, celo, train_dl, val_dl, seq_len, device,
                        target_val_loss, max_steps, eval_every, args, use_amp=False):
    model.train()
    crit = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    start = time.perf_counter()
    steps = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        
    ema = Smoothed(alpha=0.98)
    last_val = float("inf")
    last_print = time.perf_counter()
    tokens_per_step = seq_len * train_dl.batch_size
    seen_tokens = 0




    celo.reset_state()
    it = iter(train_dl)
    while steps < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0,1)

        for p in model.parameters(): p.grad = None

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                logits = model(src, src_mask=mask)
                loss = crit(logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
                ema_loss = ema.update(float(loss.item()))

            scaler.scale(loss).backward()
            scaler.unscale_(None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # CeLO is @no_grad, fine under autocast as long as weights/buffers match dtype
            celo.step_inplace(list(model.parameters()), loss_value=float(loss.item()), step_index=steps+1)
            scaler.step(torch.optim.SGD([torch.zeros(1, device=device)], lr=1.0))  # dummy step to satisfy scaler
            scaler.update()
        else:
            logits = model(src, src_mask=mask)
            loss = crit(logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
            ema_loss = ema.update(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            celo.step_inplace(list(model.parameters()), loss_value=float(loss.item()), step_index=steps+1)

        steps += 1

        seen_tokens += tokens_per_step
        if (steps % args.log_every) == 0:
            now = time.perf_counter()
            dt = now - last_print
            ips = (args.log_every * tokens_per_step) / max(dt, 1e-9)  # tokens/sec
            mem = ""
            if args.print_gpu_mem and torch.cuda.is_available():
                mem = f" | GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB max"
                torch.cuda.reset_peak_memory_stats()
            print(f"[CeLO] step {steps:5d} | train_loss {loss.item():.4f} | ema {ema_loss:.4f} "
                  f"| val {last_val:.4f} | tok/s {ips:,.0f} | elapsed {fmt_secs(time.perf_counter()-start)}{mem}",
                  flush=True)
            last_print = now


        if (steps % eval_every) == 0:
            v = eval_val_loss(model, val_dl, seq_len, device)
            last_val = v
            print(f"[CeLO][eval] step {steps:5d} | val_loss {v:.4f} | target {target_val_loss:.4f}",
                  flush=True)
            if v <= target_val_loss:
                print(f"[CeLO][done] step {steps} | time {fmt_secs(time.perf_counter()-start)} | val {v:.4f}",
                      flush=True)
                return time.perf_counter() - start, steps, v


    return None, steps, eval_val_loss(model, val_dl, seq_len, device)

def time_to_target_torchopt(model, optimizer, train_dl, val_dl, seq_len, device,
                            target_val_loss, max_steps, eval_every, args):
    model.train()
    crit = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    start = time.perf_counter()
    steps = 0
    it = iter(train_dl)

    ema = Smoothed(alpha=0.98)
    last_val = float("inf")
    last_print = time.perf_counter()
    tokens_per_step = seq_len * train_dl.batch_size
    seen_tokens = 0


    while steps < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0,1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, src_mask=mask)
        loss = crit(logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
        ema_loss = ema.update(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        steps += 1

        seen_tokens += tokens_per_step
        if (steps % args.log_every) == 0:
            now = time.perf_counter()
            dt = now - last_print
            ips = (args.log_every * tokens_per_step) / max(dt, 1e-9)
            mem = ""
            if args.print_gpu_mem and torch.cuda.is_available():
                mem = f" | GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB max"
                torch.cuda.reset_peak_memory_stats()
            print(f"[TorchOpt] step {steps:5d} | train_loss {loss.item():.4f} | ema {ema_loss:.4f} "
                  f"| val {last_val:.4f} | tok/s {ips:,.0f} | elapsed {fmt_secs(time.perf_counter()-start)}{mem}",
                  flush=True)
            last_print = now

        if (steps % eval_every) == 0:
            v = eval_val_loss(model, val_dl, seq_len, device)
            last_val = v
            print(f"[TorchOpt][eval] step {steps:5d} | val_loss {v:.4f} | target {target_val_loss:.4f}",
                  flush=True)
            if v <= target_val_loss:
                print(f"[TorchOpt][done] step {steps} | time {fmt_secs(time.perf_counter()-start)} | val {v:.4f}",
                      flush=True)
                return time.perf_counter() - start, steps, v


    return None, steps, eval_val_loss(model, val_dl, seq_len, device)

def time_to_target_pylo(model, optimizer, train_dl, val_dl, seq_len, device,
                        target_val_loss, max_steps, eval_every, args):
    """
    PyLO/VeLO contract: optimizer.step(loss) (not step()).
    """
    model.train()
    crit = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    start = time.perf_counter()
    steps = 0
    it = iter(train_dl)

    ema = Smoothed(alpha=0.98)
    last_val = float("inf")
    last_print = time.perf_counter()
    tokens_per_step = seq_len * train_dl.batch_size

    while steps < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        src = x.transpose(0,1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, src_mask=mask)
        loss = crit(logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
        ema_loss = ema.update(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(loss)   # <-- PyLO / VeLO uses .step(loss)
        steps += 1

        if (steps % args.log_every) == 0:
            now = time.perf_counter()
            dt = now - last_print
            ips = (args.log_every * tokens_per_step) / max(dt, 1e-9)
            mem = ""
            if args.print_gpu_mem and torch.cuda.is_available():
                mem = f" | GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB max"
                torch.cuda.reset_peak_memory_stats()
            print(f"[VeLO] step {steps:5d} | train_loss {loss.item():.4f} | ema {ema_loss:.4f} "
                  f"| val {last_val:.4f} | tok/s {ips:,.0f} | elapsed {fmt_secs(time.perf_counter()-start)}{mem}",
                  flush=True)
            last_print = now

        if (steps % eval_every) == 0:
            v = eval_val_loss(model, val_dl, seq_len, device)
            last_val = v
            print(f"[VeLO][eval] step {steps:5d} | val_loss {v:.4f} | target {target_val_loss:.4f}", flush=True)
            if v <= target_val_loss:
                print(f"[VeLO][done] step {steps} | time {fmt_secs(time.perf_counter()-start)} | val {v:.4f}", flush=True)
                return time.perf_counter() - start, steps, v

    return None, steps, eval_val_loss(model, val_dl, seq_len, device)

def build_adafactor(params, lr):
    try:
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
    except Exception:
        warnings.warn("transformers Adafactor not found; falling back to AdamW for the 'Adafactor' slot.")
        return torch.optim.AdamW(params, lr=lr)


# -------------------------------
# VeLO compression utilities
# -------------------------------
def _modules_linear(m):
    return {nn.Linear}

def apply_velo_quantization_fp16(velo):
    velo.half()   # convert learned optimizer nets to fp16
    return velo

def apply_velo_dynamic_int8_linear(velo):
    try:
        from torch.ao.quantization import quantize_dynamic
        q_velo = quantize_dynamic(velo, _modules_linear(velo), dtype=torch.qint8)
        return q_velo
    except Exception as e:
        warnings.warn(f"Dynamic int8 quantization for VeLO unavailable: {e}")
        return velo

def apply_velo_pruning_l1(velo, amount: float):
    # Walk submodules; prune Linear weights (and other obvious fully-connected heads)
    for m in velo.modules():
        if isinstance(m, nn.Linear):
            try:
                prune.l1_unstructured(m, name="weight", amount=amount)
            except Exception as e:
                warnings.warn(f"Prune failed for {m}: {e}")
    return velo

# -------------------------------
# Distillation (optional, separate)
# -------------------------------
def distill_celo_teacher_student(teacher: CeLOLite, student: CeLOLite,
                                 model_template, train_dl, device, seq_len,
                                 steps=100, lr=1e-3):
    """
    Lightweight distillation: student minimizes MSE between teacher- and student-proposed updates
    using CeLOMetaAdapter on identical gradient streams (first-order, no higher-order graphs).
    """
    teacher_meta = CeLOMetaAdapter(teacher).to(device)
    student_meta = CeLOMetaAdapter(student).to(device)
    opt = torch.optim.AdamW(student_meta.parameters(), lr=lr)
    crit = nn.MSELoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    it = iter(train_dl)
    for k in range(1, steps+1):
        try: x, y = next(it)
        except StopIteration: it = iter(train_dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        # fresh theta each step to avoid bias
        model = model_template().to(device)
        base = {n:p for n,p in model.named_parameters()}
        theta = {n:p.detach().clone().requires_grad_(True) for n,p in base.items()}

        # forward/backward to get grads
        logits = model(x.transpose(0,1), src_mask=mask)
        loss = nn.functional.cross_entropy(
            logits.transpose(0,1).reshape(-1, logits.size(-1)), y.view(-1))
        names = list(theta.keys())
        grads_list = torch.autograd.grad(loss, [theta[n] for n in names], retain_graph=False, create_graph=False)
        grads = {n:g for n,g in zip(names, grads_list)}

        # teacher targets
        up_t, state_t = teacher_meta.propose_updates(theta, grads, state={}, step_index=k, loss_value=loss.detach(), first_order=True)
        # student prediction
        up_s, state_s = student_meta.propose_updates(theta, grads, state={}, step_index=k, loss_value=loss.detach(), first_order=True)

        # MSE over concatenated parameter updates
        def cat(upd_dict): return torch.cat([u.reshape(-1) for u in upd_dict.values() if u is not None])
        target, pred = cat(up_t), cat(up_s)
        opt.zero_grad(set_to_none=True)
        l = crit(pred, target.detach())
        l.backward()
        torch.nn.utils.clip_grad_norm_(student_meta.parameters(), 1.0)
        opt.step()
    return student  # student now distilled

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celo_ckpt", type=str, default=None, help="path to base CeLO checkpoint from train_celo_base.py")
    ap.add_argument("--data_path", type=str, default="enwik8")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Benchmark knobs
    ap.add_argument("--target_val_loss", type=float, default=2.30)  # set your target
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sgd_momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2025)

    # Which runs to include
    ap.add_argument("--run_adamw", action="store_true")
    ap.add_argument("--run_adafactor", action="store_true")
    ap.add_argument("--run_sgd", action="store_true")
    ap.add_argument("--run_celo_base", action="store_true")
    ap.add_argument("--run_celo_quant_fp16", action="store_true")
    ap.add_argument("--run_celo_quant_int8_linear", action="store_true")
    ap.add_argument("--run_celo_prune", action="store_true")
    ap.add_argument("--prune_amount", type=float, default=0.3)  # 30% L1 pruning

    # Distillation (kept separate per your plan; off by default)
    ap.add_argument("--run_celo_distill", action="store_true")
    ap.add_argument("--distill_steps", type=int, default=0)
    ap.add_argument("--student_hidden_sched", type=int, default=16)
    ap.add_argument("--student_hidden_rule", type=int, default=16)

    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--print_gpu_mem", action="store_true")

    ap.add_argument("--results_csv", type=str, default="time_to_target.csv")

    # VeLO (PyLO) toggles
    ap.add_argument("--velo_ckpt", type=str, default=None, help="optional VeLO checkpoint (state_dict or full)")
    ap.add_argument("--run_velo", action="store_true")
    ap.add_argument("--run_velo_quant_fp16", action="store_true")
    ap.add_argument("--run_velo_quant_int8_linear", action="store_true")
    ap.add_argument("--run_velo_prune", action="store_true")
    ap.add_argument("--velo_prune_amount", type=float, default=0.3)

    ap.add_argument("--use_celo_hf", action="store_true",
                    help="Use pretrained CeLO (JAX) via Hugging Face (amoudgl/celo)")
    ap.add_argument("--celo_hf_repo", type=str, default="amoudgl/celo",
                    help="Hugging Face repo id for CeLO checkpoint")
    ap.add_argument("--celo_hf_file", type=str, default="theta.state",
                    help="Checkpoint filename inside the HF repo")
    ap.add_argument("--celo_hf_steps", type=int, default=3000,
                    help="num_steps argument for CeLO.init(...)")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_dl, val_dl = build_data(args.data_path, args.seq_len, args.batch_size)
    def model_template(): return make_model(args.d_model, args.nhead, args.nlayers, args.dim_ff, device)

    need_celo = any([
        args.run_celo_base,
        args.run_celo_quant_fp16,
        args.run_celo_quant_int8_linear,
        args.run_celo_prune,
        args.run_celo_distill,
    ])
    sd = None
    if need_celo:
        if not args.celo_ckpt:
            raise FileNotFoundError("--celo_ckpt is required when running a CeLO mode.")
        base_celo = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        sd = torch.load(args.celo_ckpt, map_location=device)
        base_celo.load_state_dict(sd)


    runs = []
    # --------- Baselines (AdamW / Adafactor / SGD-momentum) ----------
    if args.run_adamw:
        m = model_template()
        t, steps, v = time_to_target_torchopt(m, torch.optim.AdamW(m.parameters(), lr=args.lr),
                                              train_dl, val_dl, args.seq_len, device,
                                              args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append(("AdamW", t, steps, v))

    if args.run_adafactor:
        m = model_template()
        opt = build_adafactor(m.parameters(), lr=args.lr)
        t, steps, v = time_to_target_torchopt(m, opt, train_dl, val_dl, args.seq_len, device,
                                              args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append(("Adafactor", t, steps, v))

    if args.run_sgd:
        m = model_template()
        opt = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.sgd_momentum)
        t, steps, v = time_to_target_torchopt(m, opt, train_dl, val_dl, args.seq_len, device,
                                              args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append(("SGD+Momentum", t, steps, v))

    # --------- VeLO (PyLO) ----------
    if (args.run_velo or args.run_velo_quant_fp16 or args.run_velo_quant_int8_linear or args.run_velo_prune):
        if not PYLO_AVAILABLE:
            raise RuntimeError("pylo (VeLO) is not available. Please install your PyLO build to use VeLO.")

    if args.run_velo:
        m = model_template()
        velo = VeLO(m.parameters(), lr=args.lr)   # PyLO constructor; adjust kwargs if your build differs
        if args.velo_ckpt:
            sd = torch.load(args.velo_ckpt, map_location=device)
            try:
                velo.load_state_dict(sd)  # preferred if checkpoint is optimizer state_dict
            except Exception:
                # If user provides {'optimizer': state_dict, ...}
                if isinstance(sd, dict) and "state_dict" in sd:
                    velo.load_state_dict(sd["state_dict"])
                else:
                    warnings.warn("Could not load VeLO checkpoint (unexpected format); running uninitialized.")
        t, steps, v = time_to_target_pylo(m, velo, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append(("VeLO", t, steps, v))

    if args.run_velo_quant_fp16:
        m = model_template()
        velo = VeLO(m.parameters(), lr=args.lr)
        if args.velo_ckpt:
            sd = torch.load(args.velo_ckpt, map_location=device)
            try:
                velo.load_state_dict(sd)
            except Exception:
                if isinstance(sd, dict) and "state_dict" in sd:
                    velo.load_state_dict(sd["state_dict"])
        velo = apply_velo_quantization_fp16(velo)
        t, steps, v = time_to_target_pylo(m, velo, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append(("VeLO (quant fp16)", t, steps, v))

    if args.run_velo_quant_int8_linear:
        m = model_template()
        velo = VeLO(m.parameters(), lr=args.lr)
        if args.velo_ckpt:
            sd = torch.load(args.velo_ckpt, map_location=device)
            try:
                velo.load_state_dict(sd)
            except Exception:
                if isinstance(sd, dict) and "state_dict" in sd:
                    velo.load_state_dict(sd["state_dict"])
        velo = apply_velo_dynamic_int8_linear(velo)
        t, steps, v = time_to_target_pylo(m, velo, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append(("VeLO (quant int8 linear)", t, steps, v))

    if args.run_velo_prune:
        m = model_template()
        velo = VeLO(m.parameters(), lr=args.lr)
        if args.velo_ckpt:
            sd = torch.load(args.velo_ckpt, map_location=device)
            try:
                velo.load_state_dict(sd)
            except Exception:
                if isinstance(sd, dict) and "state_dict" in sd:
                    velo.load_state_dict(sd["state_dict"])
        velo = apply_velo_pruning_l1(velo, amount=args.velo_prune_amount)
        t, steps, v = time_to_target_pylo(m, velo, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append((f"VeLO (pruned {args.velo_prune_amount:.2f})", t, steps, v))

    # --------- CeLO base ----------
    if args.run_celo_base:
        m = model_template()
        ce = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        ce.load_state_dict(sd)
        t, steps, v = time_to_target_celo(m, ce, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args, use_amp=False)
        runs.append(("CeLO (base)", t, steps, v))

    # --------- CeLO quantization (fp16 autocast) ----------
    if args.run_celo_quant_fp16:
        m = model_template()
        ce = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        ce.load_state_dict(sd)
        ce, info = apply_quantization_fp16(ce)
        t, steps, v = time_to_target_celo(m, ce, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args, use_amp=info.get("use_amp", False))
        runs.append(("CeLO (quant fp16)", t, steps, v))

    # --------- CeLO dynamic int8 on Linear (CPU-friendly) ----------
    if args.run_celo_quant_int8_linear:
        m = model_template()
        ce = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        ce.load_state_dict(sd)
        ce, info = apply_dynamic_int8_linear_only(ce)
        t, steps, v = time_to_target_celo(m, ce, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args, use_amp=False)
        runs.append(("CeLO (quant int8 linear)", t, steps, v))

    # --------- CeLO pruning ----------
    if args.run_celo_prune:
        m = model_template()
        ce = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        ce.load_state_dict(sd)
        ce = apply_pruning_l1(ce, amount=args.prune_amount)
        t, steps, v = time_to_target_celo(m, ce, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every,args, use_amp=False)
        runs.append((f"CeLO (pruned {args.prune_amount:.2f})", t, steps, v))

    # --------- CeLO distillation (optional; separate) ----------
    if args.run_celo_distill:
        m = model_template()
        teacher = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        teacher.load_state_dict(sd)
        student = CeLOLite(hidden_sched=args.student_hidden_sched, hidden_rule=args.student_hidden_rule,
                           alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        if args.distill_steps > 0:
            student = distill_celo_teacher_student(teacher, student, model_template, train_dl, device, args.seq_len,
                                                   steps=args.distill_steps, lr=1e-3)
        t, steps, v = time_to_target_celo(m, student, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every,args, use_amp=False)
        runs.append((f"CeLO (distilled hs={args.student_hidden_sched}/hr={args.student_hidden_rule})", t, steps, v))

    # --------- CeLO pretrained from Hugging Face (JAX) ----------
    if args.use_celo_hf:
        m = model_template()
        celo_hf = CeloJaxAdapter(
            m,
            repo=args.celo_hf_repo,
            filename=args.celo_hf_file,
            num_steps=args.celo_hf_steps
        )
        t, steps, v = time_to_target_celo_hf(
            m, celo_hf, train_dl, val_dl, args.seq_len, device,
            args.target_val_loss, args.max_steps, args.eval_every, args
        )
        runs.append(("CeLO (pretrained HF, JAX bridge)", t, steps, v))

    # save results
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    with open(args.results_csv, "w") as f:
        f.write("method,seconds_to_target,steps_to_target,final_val_loss\n")
        for (name, t, steps, v) in runs:
            sec = -1 if t is None else t
            f.write(f"{name},{sec},{steps},{v:.6f}\n")
    print("\n== Time-to-target results ==")
    for r in runs: print(r)

if __name__ == "__main__":
    main()
