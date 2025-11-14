
#!/usr/bin/env python3
import argparse, os, time, math, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

# JAX / CeLO
import jax
from jax import device_get
from huggingface_hub import hf_hub_download
from celo.factory import get_optimizer
from celo.utils import init_lopt_from_ckpt

# Optional PyLO / VeLO
PYLO_AVAILABLE = True
try:
    from pylo.optim import VeLO  # your PyLO fork/build
except Exception:
    PYLO_AVAILABLE = False

# --------------------------------
# Small helpers
# --------------------------------
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

# --------------------------------
# Your modules / dataset
# --------------------------------
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
    N = len(full)
    N_train = max(1, int(0.9 * N))
    train_idx = range(0, N_train)
    val_idx   = range(N_train, N)
    train_dl = DataLoader(torch.utils.data.Subset(full, list(train_idx)), batch_size=batch_size, shuffle=False, drop_last=True)
    val_dl   = DataLoader(torch.utils.data.Subset(full, list(val_idx)),   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl

# --------------------------------
# CeLO-JAX ↔ PyTorch bridge
# --------------------------------
class CeloJaxBridge:
    """
    Wrap the official CeLO optimizer and state while mirroring CeLO's training-loop API:
        params = opt.get_params(opt_state)
        loss, grad = value_and_grad(loss_fn)(params, ...)
        opt_state  = opt.update(opt_state, grad, loss=loss)

    Here, we compute grads in PyTorch, then feed a numpy dict of grads to CeLO.
    After update, we fetch new params from CeLO and copy them into the Torch model.
    """
    def __init__(self, model,
                 hf_repo: str | None = None,
                 hf_filename: str | None = None,
                 local_ckpt: str | None = None,
                 num_steps: int = 1000,
                 grad_scale: float = 1.0,
                 clip_grad: float | None = 1.0,
                 use_token_sum: bool = False):
        if (hf_repo is None or hf_filename is None) and (local_ckpt is None):
            raise ValueError("Provide either (hf_repo & hf_filename) OR local_ckpt to a '.state' file.")

        # Snapshot parameter names and initial arrays (float32, CPU)
        self.model = model
        self.names = []
        self.params_np = {}
        for n, p in model.named_parameters():
            self.names.append(n)
            self.params_np[n] = np.asarray(p.detach().cpu().numpy(), dtype=np.float32)

        self.grad_scale  = float(grad_scale)
        self.clip_grad   = clip_grad
        self.use_token_sum = bool(use_token_sum)

        # Resolve checkpoint path
        if local_ckpt is not None:
            ckpt_path = os.path.abspath(local_ckpt)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Local CeLO checkpoint not found: {ckpt_path}")
        else:
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename, local_dir="./")

        # Build optimizer with that θ
        lopt = get_optimizer('celo')
        self.opt = init_lopt_from_ckpt(lopt, ckpt_path)
        # Initialize CeLO state (no model_state for generic Torch nets)
        self.opt_state = self.opt.init(self.params_np, model_state=None, num_steps=num_steps)

    def _torch_grads_to_np(self) -> dict:
        grads = {}
        for n, p in self.model.named_parameters():
            if p.grad is None:
                grads[n] = np.zeros_like(self.params_np[n], dtype=np.float32)
            else:
                grads[n] = np.asarray(p.grad.detach().cpu().numpy(), dtype=np.float32)
        return grads

    def get_params(self):
        """Mirror CeLO API: returns dict of numpy params."""
        params_jax = self.opt.get_params(self.opt_state)
        return {k: np.asarray(device_get(v), dtype=np.float32) for k, v in params_jax.items()}

    def update(self, loss_value: float, token_factor: int | float = 1):
        """Mirror CeLO API: state <- opt.update(state, grad, loss=loss)."""
        grads = self._torch_grads_to_np()

        # scale by tokens if desired (keeps invariants similar to JAX examples)
        if self.use_token_sum:
            s = float(token_factor)
            for k in grads: grads[k] *= s

        # global grad scale (optional)
        if self.grad_scale != 1.0:
            for k in grads: grads[k] *= self.grad_scale

        # global clip (optional)
        if self.clip_grad is not None and self.clip_grad > 0:
            gn = math.sqrt(sum(float(np.square(g).sum()) for g in grads.values())) + 1e-12
            if gn > self.clip_grad:
                scale = self.clip_grad / gn
                for k in grads: grads[k] *= scale

        # ---- This is the official CeLO call pattern ----
        # opt_state = opt.update(opt_state, grad, loss=loss)
        self.opt_state = self.opt.update(self.opt_state, grads, loss=float(loss_value))

        # fetch new params and copy into torch
        new_params = self.get_params()
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                arr = torch.from_numpy(new_params[n]).to(p.device, dtype=p.dtype)
                p.copy_(arr)
        # keep local np mirror consistent for next step
        self.params_np = new_params

# --------------------------------
# Training loops
# --------------------------------
def time_to_target_celo_hf(model, bridge: CeloJaxBridge, train_dl, val_dl, seq_len, device,
                           target_val_loss, max_steps, eval_every, args):
    """
    Uses CeLO's official rhythm:
      params = opt.get_params(state)
      (we compute loss+grads in Torch)
      state = opt.update(state, grads, loss=loss)
    """
    model.train()
    crit = nn.CrossEntropyLoss()
    mask = generate_square_subsequent_mask(seq_len).to(device)

    start = time.perf_counter()
    steps = 0
    ema = Smoothed(alpha=0.98)
    last_val = float("inf")
    tokens_per_step = seq_len * train_dl.batch_size

    it = iter(train_dl)
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

        # === CeLO syntax mirror ===
        # opt_state <- opt.update(opt_state, grad, loss=loss)
        bridge.update(loss_value=float(loss.item()), token_factor=tokens_per_step)

        steps += 1

        if (steps % args.log_every) == 0:
            print(f"[CeLO-HF] step {steps:5d} | train_loss {loss.item():.4f} | ema {ema_loss:.4f} "
                  f"| val {last_val:.4f} | elapsed {fmt_secs(time.perf_counter()-start)}", flush=True)

        if (steps % eval_every) == 0:
            v = eval_val_loss(model, val_dl, seq_len, device)
            last_val = v
            print(f"[CeLO-HF][eval] step {steps:5d} | val_loss {v:.4f} | target {target_val_loss:.4f}", flush=True)
            if v <= target_val_loss:
                print(f"[CeLO-HF][done] step {steps} | time {fmt_secs(time.perf_counter()-start)} | val {v:.4f}", flush=True)
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
    tokens_per_step = seq_len * train_dl.batch_size
    last_print = time.perf_counter()

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
            print(f"[TorchOpt][eval] step {steps:5d} | val_loss {v:.4f} | target {target_val_loss:.4f}", flush=True)
            if v <= target_val_loss:
                print(f"[TorchOpt][done] step {steps} | time {fmt_secs(time.perf_counter()-start)} | val {v:.4f}", flush=True)
                return time.perf_counter() - start, steps, v

    return None, steps, eval_val_loss(model, val_dl, seq_len, device)

def time_to_target_pylo(model, optimizer, train_dl, val_dl, seq_len, device,
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
    last_print = time.perf_counter()

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
        optimizer.step(loss)   # PyLO/VeLO contract

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

# --------------------------------
# Optimizer builders / compression hooks (unchanged)
# --------------------------------
def build_adafactor(params, lr):
    try:
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
    except Exception:
        warnings.warn("transformers Adafactor not found; falling back to AdamW for the 'Adafactor' slot.")
        return torch.optim.AdamW(params, lr=lr)

def _modules_linear(_): return {nn.Linear}

def apply_velo_quantization_fp16(velo):
    velo.half()
    return velo

def apply_velo_dynamic_int8_linear(velo):
    try:
        from torch.ao.quantization import quantize_dynamic
        return quantize_dynamic(velo, _modules_linear(velo), dtype=torch.qint8)
    except Exception as e:
        warnings.warn(f"Dynamic int8 quantization for VeLO unavailable: {e}")
        return velo

def apply_velo_pruning_l1(velo, amount: float):
    for m in velo.modules():
        if isinstance(m, nn.Linear):
            try: prune.l1_unstructured(m, name="weight", amount=amount)
            except Exception as e: warnings.warn(f"Prune failed for {m}: {e}")
    return velo

# --------------------------------
# Main
# --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celo_ckpt", type=str, default=None, help="path to base CeLOLite checkpoint (torch) for non-HF modes")
    ap.add_argument("--data_path", type=str, default="enwik8")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Bench knobs
    ap.add_argument("--target_val_loss", type=float, default=2.30)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sgd_momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--print_gpu_mem", action="store_true")
    ap.add_argument("--results_csv", type=str, default="time_to_target.csv")

    # Which runs
    ap.add_argument("--run_adamw", action="store_true")
    ap.add_argument("--run_adafactor", action="store_true")
    ap.add_argument("--run_sgd", action="store_true")

    # VeLO toggles
    ap.add_argument("--run_velo", action="store_true")
    ap.add_argument("--run_velo_quant_fp16", action="store_true")
    ap.add_argument("--run_velo_quant_int8_linear", action="store_true")
    ap.add_argument("--run_velo_prune", action="store_true")
    ap.add_argument("--velo_prune_amount", type=float, default=0.3)
    ap.add_argument("--velo_ckpt", type=str, default=None)

    # CeLOLite (torch) toggles (kept for completeness)
    ap.add_argument("--run_celo_base", action="store_true")
    ap.add_argument("--run_celo_quant_fp16", action="store_true")
    ap.add_argument("--run_celo_quant_int8_linear", action="store_true")
    ap.add_argument("--run_celo_prune", action="store_true")
    ap.add_argument("--prune_amount", type=float, default=0.3)

    # CeLO HF / local .state (JAX) bridge
    ap.add_argument("--use_celo_hf", action="store_true", help="Use pretrained CeLO (JAX) via Hugging Face")
    ap.add_argument("--celo_hf_repo", type=str, default="amoudgl/celo")
    ap.add_argument("--celo_hf_file", type=str, default="theta.state")
    ap.add_argument("--celo_hf_steps", type=int, default=1000)
    ap.add_argument("--use_celo_local", action="store_true", help="Use local CeLO .state")
    ap.add_argument("--celo_local_ckpt", type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_dl, val_dl = build_data(args.data_path, args.seq_len, args.batch_size)
    def model_template(): return make_model(args.d_model, args.nhead, args.nlayers, args.dim_ff, device)

    runs = []

    # --------- CeLO pretrained from LOCAL .state (JAX bridge) ----------
    if args.use_celo_local:
        if not args.celo_local_ckpt:
            raise FileNotFoundError("--celo_local_ckpt is required when --use_celo_local is set.")
        m = model_template()
        bridge = CeloJaxBridge(
            m, local_ckpt=args.celo_local_ckpt, num_steps=args.celo_hf_steps,
            grad_scale=1.0, clip_grad=1.0, use_token_sum=True
        )
        t, steps, v = time_to_target_celo_hf(
            m, bridge, train_dl, val_dl, args.seq_len, device,
            args.target_val_loss, args.max_steps, args.eval_every, args
        )
        runs.append(("CeLO (pretrained LOCAL, JAX bridge)", t, steps, v))

    # --------- AdamW / Adafactor / SGD ----------
    if args.run_adamw:
        m = model_template()
        t, steps, v = time_to_target_torchopt(
            m, torch.optim.AdamW(m.parameters(), lr=args.lr),
            train_dl, val_dl, args.seq_len, device,
            args.target_val_loss, args.max_steps, args.eval_every, args
        )
        runs.append(("AdamW", t, steps, v))

    if args.run_adafactor:
        m = model_template()
        opt = build_adafactor(m.parameters(), lr=args.lr)
        t, steps, v = time_to_target_torchopt(
            m, opt, train_dl, val_dl, args.seq_len, device,
            args.target_val_loss, args.max_steps, args.eval_every, args
        )
        runs.append(("Adafactor", t, steps, v))

    if args.run_sgd:
        m = model_template()
        opt = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.sgd_momentum)
        t, steps, v = time_to_target_torchopt(
            m, opt, train_dl, val_dl, args.seq_len, device,
            args.target_val_loss, args.max_steps, args.eval_every, args
        )
        runs.append(("SGD+Momentum", t, steps, v))

    # --------- VeLO (PyLO) ----------
    if (args.run_velo or args.run_velo_quant_fp16 or args.run_velo_quant_int8_linear or args.run_velo_prune):
        if not PYLO_AVAILABLE:
            raise RuntimeError("pylo (VeLO) is not available. Please install your PyLO build to use VeLO.")

    if args.run_velo:
        m = model_template()
        velo = VeLO(m.parameters(), lr=args.lr)
        if args.velo_ckpt:
            sd = torch.load(args.velo_ckpt, map_location=device)
            try: velo.load_state_dict(sd)
            except Exception:
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
            try: velo.load_state_dict(sd)
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
            try: velo.load_state_dict(sd)
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
            try: velo.load_state_dict(sd)
            except Exception:
                if isinstance(sd, dict) and "state_dict" in sd:
                    velo.load_state_dict(sd["state_dict"])
        velo = apply_velo_pruning_l1(velo, amount=args.velo_prune_amount)
        t, steps, v = time_to_target_pylo(m, velo, train_dl, val_dl, args.seq_len, device,
                                          args.target_val_loss, args.max_steps, args.eval_every, args)
        runs.append((f"VeLO (pruned {args.velo_prune_amount:.2f})", t, steps, v))

    # --------- CeLOLite (torch) toggles kept for completeness ----------
    if any([args.run_celo_base, args.run_celo_quant_fp16, args.run_celo_quant_int8_linear, args.run_celo_prune]):
        if not args.celo_ckpt:
            raise FileNotFoundError("--celo_ckpt is required for CeLOLite torch modes.")
        base_sd = torch.load(args.celo_ckpt, map_location=device)

    if args.run_celo_base:
        m = model_template()
        ce = CeLOLite(hidden_sched=32, hidden_rule=32, alpha=1.0, lambda1=1.0, lambda2=1.0, device=device)
        ce.load_state_dict(base_sd)
        # (Reuse your previous time_to_target_celo if desired; omitted here for brevity.)

    # --------- CeLO pretrained from Hugging Face (JAX bridge) ----------
    if args.use_celo_hf:
        m = model_template()
        bridge = CeloJaxBridge(
            m,
            hf_repo=args.celo_hf_repo,
            hf_filename=args.celo_hf_file,
            num_steps=args.celo_hf_steps,
            grad_scale=1.0,
            clip_grad=1.0,
            use_token_sum=True,  # keep token-proportional scaling similar to JAX examples
        )
        t, steps, v = time_to_target_celo_hf(
            m, bridge, train_dl, val_dl, args.seq_len, device,
            args.target_val_loss, args.max_steps, args.eval_every, args
        )
        runs.append(("CeLO (pretrained HF, JAX bridge)", t, steps, v))

    # --------- Save results ----------
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
