#!/usr/bin/env python3
"""Verify EP Student base_jit.pt accepts (obs[753], depth_latent[32]) -> action[12]."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
jit = root / "policy/go2/go2_ep_student/base_jit.pt"
m = torch.jit.load(str(jit), map_location="cpu")
m.eval()
obs = torch.zeros(1, 753)
latent = torch.zeros(1, 32)
with torch.no_grad():
    out = m(obs, latent)
out = out.reshape(-1)
assert out.numel() == 12, out.shape
assert torch.isfinite(out).all()
print(f"SMOKE_OK_JIT dual-input shape={tuple(out.shape)} sample={out[:4].tolist()}")
