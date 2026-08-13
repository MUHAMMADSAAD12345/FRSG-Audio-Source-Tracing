#!/usr/bin/env python3
"""Home smoke test: dual-branch model + losses without MLAAD/GPU.

Creates tiny fake CORES/XLSR caches, runs a few training steps, saves a ckpt,
and checks shapes / finite losses. Not for paper metrics.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class ExpertMLP(nn.Module):
  def __init__(self, in_dim, hidden=512, out=256, dropout=0.3):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(hidden, out), nn.BatchNorm1d(out), nn.ReLU(inplace=True), nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class GatingNetwork(nn.Module):
  def __init__(self, in_dim=512, hidden=128, dropout=0.2):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 2))

  def forward(self, e_hc, e_ssl):
    return torch.softmax(self.net(torch.cat([e_hc, e_ssl], dim=-1)), dim=-1)


class DualBranchModel(nn.Module):
  def __init__(self, cores_dim=66, ssl_dim=1024, num_classes=24):
    super().__init__()
    self.expert_hc = ExpertMLP(cores_dim)
    self.expert_ssl = ExpertMLP(ssl_dim)
    self.gate = GatingNetwork()
    self.classifier = nn.Linear(256, num_classes)

  def forward(self, x_hc, x_ssl):
    e_hc = self.expert_hc(x_hc)
    e_ssl = self.expert_ssl(x_ssl)
    alpha = self.gate(e_hc, e_ssl)
    e_fused = alpha[:, 0:1] * e_hc + alpha[:, 1:2] * e_ssl
    return self.classifier(e_fused), alpha


def energy_from_logits(logits):
  return -torch.logsumexp(logits, dim=-1)


def main() -> int:
  device = torch.device('cpu')
  n, n_class = 64, 24
  rng = np.random.RandomState(0)
  x_hc = rng.randn(n, 66).astype(np.float32)
  x_ssl = rng.randn(n, 1024).astype(np.float32)
  y = rng.randint(0, n_class, size=n).astype(np.int64)
  is_ood = np.zeros(n, dtype=bool)
  is_ood[n // 2 :] = True
  y[is_ood] = -1

  model = DualBranchModel().to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

  xh = torch.from_numpy(x_hc)
  xs = torch.from_numpy(x_ssl)
  yt = torch.from_numpy(y)
  ood = torch.from_numpy(is_ood)

  model.train()
  for step in range(3):
    opt.zero_grad()
    logits, alpha = model(xh, xs)
    assert logits.shape == (n, n_class)
    assert alpha.shape == (n, 2)
    assert torch.allclose(alpha.sum(-1), torch.ones(n), atol=1e-5)

    id_mask = (~ood) & (yt >= 0)
    ce = F.cross_entropy(logits[id_mask], yt[id_mask])
    E = energy_from_logits(logits)
    energy = F.relu(E[~ood] - (-15.0)).mean() + F.relu((-2.0) - E[ood]).mean()
    a_id = alpha[~ood].mean(0).clamp_min(1e-8)
    a_ood = alpha[ood].mean(0).clamp_min(1e-8)
    gate = -torch.sum(a_id * (torch.log(a_id) - torch.log(a_ood)))
    ent = torch.sum(alpha.clamp_min(1e-8) * torch.log(alpha.clamp_min(1e-8)), dim=-1).mean()
    loss = ce + 0.5 * energy + 0.05 * gate + 0.3 * ent
    assert torch.isfinite(loss), loss
    loss.backward()
    opt.step()
    print(
        f'step {step}: loss={float(loss.detach()):.4f} ce={float(ce.detach()):.4f} '
        f'alpha_ssl_mean={float(alpha[:,1].mean().detach()):.3f}'
    )

  with tempfile.TemporaryDirectory() as td:
    ckpt = Path(td) / 'dual_branch_best.pt'
    torch.save({'model': model.state_dict()}, ckpt)
    loaded = DualBranchModel()
    loaded.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    loaded.eval()
    with torch.no_grad():
      logits, alpha = loaded(xh[:4], xs[:4])
    print('reload ok', logits.shape, alpha.shape)

  print('SMOKE TEST PASSED')
  return 0


if __name__ == '__main__':
  sys.exit(main())
