# celo/optimizers/cam.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCAM(nn.Module):
    """
    Compact Associative Memory (CAM) temporal encoder (Sec. 3.3).
    Implements:
      q = WQ ξ, k = WK ξ, v = WV ξ             (Eq. 2)
      N_{t+1} = e^{-τ} N_t + φ(k_{t+1}) v_{t+1}^T
      Ψ_{t+1} = e^{-τ} Ψ_t + φ(k_{t+1})        (Eq. 4)
      Δξ = N_t^T φ(q) / (φ(q)^T Ψ_t)           (Eq. 5)
    """
    def __init__(self, d: int, n_proj: int, r_feats: int, tau: float = 0.0, use_bias: bool = False, device=None):
        """
        Args:
          d: latent width of ξ (and v)
          n_proj: dimensionality N of q,k space (WQ, WK map d -> N)
          r_feats: number of random features r
          tau: exponential discount, τ >= 0 (τ=0 => no discount)
        """
        super().__init__()
        self.d = d
        self.n_proj = n_proj
        self.r = r_feats
        self.tau = tau
        self.decay = math.exp(-tau)

        self.WQ = nn.Linear(d, n_proj, bias=use_bias)
        self.WK = nn.Linear(d, n_proj, bias=use_bias)
        self.WV = nn.Linear(d, d,      bias=use_bias)

        # Random feature params for softmax kernel (hyperbolic-cosine RF; Appendix A.1)
        # ω ~ N(0, I), then φ_HF+(z) = γ * concat(exp(ω^T z), exp(-ω^T z)) with antithetic pairing,
        # scaled by γ = 1/sqrt(r) * exp(-||z||^2 / 2). We keep the common exp-shift for stability.
        assert r_feats % 2 == 0, "r_feats must be even for antithetic pairing"
        half = r_feats // 2
        device = device if device is not None else torch.device('cpu')
        self.omega = nn.Parameter(torch.randn(half, n_proj, device=device), requires_grad=False)

        # Hidden state buffers (N_t ∈ R^{r×d}, Ψ_t ∈ R^{r})
        self.register_buffer("N_t", torch.zeros(r_feats, d, device=device))
        self.register_buffer("Psi_t", torch.zeros(r_feats,    device=device))

    @torch.no_grad()
    def reset_state(self):
        self.N_t.zero_()
        self.Psi_t.zero_()

    def _phi(self, z: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic-cosine random features for softmax kernel (stable).
        z: (..., N)
        returns: (..., r)
        """
        # shape: (..., half)
        proj = F.linear(z, self.omega)  # ω_i^T z
        # numeric stabilization: subtract max across last dim before exp
        m = proj.abs().max(dim=-1, keepdim=True).values
        exp_pos = torch.exp(proj - m)
        exp_neg = torch.exp(-proj - m)
        # γ = 1/sqrt(r) * exp(-||z||^2 / 2) * exp(m)  (we re-multiply exp(m) to undo the shift)
        gamma = (1.0 / math.sqrt(self.r)) * torch.exp(-0.5 * (z * z).sum(dim=-1, keepdim=True) + m.squeeze(-1))
        phi = torch.cat([exp_pos, exp_neg], dim=-1) * gamma
        return phi  # (..., r)

    @torch.no_grad()
    def update_state(self, k: torch.Tensor, v: torch.Tensor):
        """
        k: (B, N)
        v: (B, d)
        Treat B (time steps for a meta-token) as a stream; fold across B.
        """
        phi_k = self._phi(k)  # (B, r)
        # Fold batch online (same result as loop, but vectorized)
        if phi_k.dim() == 1:  # (r,)
            self.N_t.mul_(self.decay).add_(torch.ger(phi_k, v))     # r×d
            self.Psi_t.mul_(self.decay).add_(phi_k)                 # r
        else:
            # Apply decay once, then accumulate discounted stream: decay^t … but Eq. (4) uses stepwise decay.
            # We apply stepwise decay by rolling:
            for i in range(phi_k.size(0)):
                self.N_t.mul_(self.decay).add_(torch.ger(phi_k[i], v[i]))
                self.Psi_t.mul_(self.decay).add_(phi_k[i])

    def forward(self, xi: torch.Tensor):
        """
        xi: (B, d) or (d,) latent(s). Returns xi' = xi + Δxi with Δxi from Eq. (5).
        Also performs the online state update using (k, v) computed from xi (causal).
        """
        single = False
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
            single = True

        q = self.WQ(xi)  # (B, N)
        k = self.WK(xi)  # (B, N)
        v = self.WV(xi)  # (B, d)

        # Update state with current (k, v) stream (Eq. 4)
        self.update_state(k, v)

        # Read: Δξ = N_t^T φ(q) / (φ(q)^T Ψ_t)  (Eq. 5)
        phi_q = self._phi(q)            # (B, r)
        num = torch.matmul(phi_q, self.N_t)      # (B, d)   == (φ(q)^T N_t)
        den = torch.matmul(phi_q, self.Psi_t)    # (B,)
        # Safe divide
        den = den.clamp_min(1e-12).unsqueeze(-1) # (B,1)
        delta = num / den                         # (B, d)
        out = xi + delta

        return out.squeeze(0) if single else out

