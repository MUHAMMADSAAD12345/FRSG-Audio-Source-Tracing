"""Local dual-branch training -- exact port of Train_Dual_Branch_Colab.ipynb.

Architecture/losses are copied verbatim from the notebook (verified against the
paper): ExpertMLP experts, input-conditioned GatingNetwork, gated fusion
e_fused = alpha0*e_hc + alpha1*e_ssl, linear classifier; label-smoothed CE
(0.15) + energy margin (m_in=-15, m_out=-2, lambda_e=0.5) with Dev-OOD as
auxiliary + gate diversity KL (lambda_g=0.05) + gate entropy (lambda_h=0.3).
Gate frozen for the first 10 epochs (optimizer rebuilt on unfreeze); AdamW
1e-4/1e-4, cosine annealing to 5e-6, grad clip 5.0. Checkpoints saved by
lowest Dev FPR95 (SME scorer), exactly the paper's criterion.

Usage:
    python scripts/train_dual_branch_local.py [--epochs 150] [--batch-size 128]
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    cores_dim: int = 66
    ssl_dim: int = 1024
    expert_hidden: int = 512
    expert_out: int = 256
    gate_hidden: int = 128
    dropout_expert: float = 0.3
    dropout_gate: float = 0.2
    num_classes: int = 24

    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    gate_freeze_epochs: int = 10
    label_smoothing: float = 0.15
    grad_clip: float = 5.0
    min_lr: float = 5e-6

    lambda_e: float = 0.5
    lambda_g: float = 0.05
    lambda_h: float = 0.3
    m_in: float = -15.0
    m_out: float = -2.0

    seed: int = 42

    cores_cache: str = str(ROOT / "cache" / "cores_features.npz")
    xlsr_cache: str = str(ROOT / "cache" / "xlsr_features.npz")
    ckpt_path: str = str(ROOT / "checkpoints" / "dual_branch_best.pt")
    hist_path: str = str(ROOT / "checkpoints" / "train_history.json")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> Dict:
    loaded = np.load(path, allow_pickle=True)
    return {k: loaded[k] for k in loaded.files}


def join_caches(cores: Dict, ssl: Dict) -> Dict:
    ssl_map = {uid: i for i, uid in enumerate(ssl["utt_ids"].tolist())}
    idxs_c, idxs_s = [], []
    for i, uid in enumerate(cores["utt_ids"].tolist()):
        if uid in ssl_map:
            idxs_c.append(i)
            idxs_s.append(ssl_map[uid])
    if not idxs_c:
        raise RuntimeError("No overlapping utt_ids between CORES and XLSR caches")
    idxs_c = np.asarray(idxs_c)
    idxs_s = np.asarray(idxs_s)
    if not np.array_equal(cores["label_ids"][idxs_c], ssl["label_ids"][idxs_s]):
        raise RuntimeError("label_ids mismatch on overlapping utt_ids")
    return {
        "utt_ids": cores["utt_ids"][idxs_c],
        "splits": cores["splits"][idxs_c],
        "label_ids": cores["label_ids"][idxs_c],
        "is_ood": cores["is_ood"][idxs_c],
        "x_hc": cores["x_hc"][idxs_c].astype(np.float32),
        "x_ssl": ssl["x_ssl"][idxs_s].astype(np.float32),
    }


class DualDataset(Dataset):
    def __init__(self, x_hc, x_ssl, y, is_ood):
        self.x_hc = torch.from_numpy(x_hc).float()
        self.x_ssl = torch.from_numpy(x_ssl).float()
        self.y = torch.from_numpy(y).long()
        self.is_ood = torch.from_numpy(is_ood.astype(np.bool_))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_hc[idx], self.x_ssl[idx], self.y[idx], self.is_ood[idx]


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
        logits = self.net(torch.cat([e_hc, e_ssl], dim=-1))
        return torch.softmax(logits, dim=-1)


class DualBranchModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.expert_hc = ExpertMLP(cfg.cores_dim, cfg.expert_hidden, cfg.expert_out, cfg.dropout_expert)
        self.expert_ssl = ExpertMLP(cfg.ssl_dim, cfg.expert_hidden, cfg.expert_out, cfg.dropout_expert)
        self.gate = GatingNetwork(cfg.expert_out * 2, cfg.gate_hidden, cfg.dropout_gate)
        self.classifier = nn.Linear(cfg.expert_out, cfg.num_classes)

    def forward(self, x_hc, x_ssl):
        e_hc = self.expert_hc(x_hc)
        e_ssl = self.expert_ssl(x_ssl)
        alpha = self.gate(e_hc, e_ssl)
        e_fused = alpha[:, 0:1] * e_hc + alpha[:, 1:2] * e_ssl
        return self.classifier(e_fused), alpha


def energy_from_logits(logits):
    return -torch.logsumexp(logits, dim=-1)


def label_smoothed_ce(logits, targets, num_classes, smoothing=0.15):
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def energy_margin_loss(logits, is_ood, m_in, m_out):
    E = energy_from_logits(logits)
    id_mask = ~is_ood
    ood_mask = is_ood
    loss = logits.new_tensor(0.0)
    if id_mask.any():
        loss = loss + F.relu(E[id_mask] - m_in).mean()
    if ood_mask.any():
        loss = loss + F.relu(m_out - E[ood_mask]).mean()
    return loss


def gate_diversity_loss(alpha, is_ood):
    id_mask = ~is_ood
    ood_mask = is_ood
    if not (id_mask.any() and ood_mask.any()):
        return alpha.new_tensor(0.0)
    a_id = alpha[id_mask].mean(dim=0).clamp_min(1e-8)
    a_ood = alpha[ood_mask].mean(dim=0).clamp_min(1e-8)
    kl = torch.sum(a_id * (torch.log(a_id) - torch.log(a_ood)))
    return -kl


def gate_entropy_loss(alpha):
    a = alpha.clamp_min(1e-8)
    return torch.sum(a * torch.log(a), dim=-1).mean()


def total_loss(logits, alpha, y, is_ood, cfg):
    id_mask = (~is_ood) & (y >= 0)
    losses = {}
    if id_mask.any():
        losses["ce"] = label_smoothed_ce(logits[id_mask], y[id_mask], cfg.num_classes, cfg.label_smoothing)
    else:
        losses["ce"] = logits.new_tensor(0.0)
    losses["energy"] = energy_margin_loss(logits, is_ood, cfg.m_in, cfg.m_out)
    losses["gate"] = gate_diversity_loss(alpha, is_ood)
    losses["ent"] = gate_entropy_loss(alpha)
    losses["total"] = (
        losses["ce"]
        + cfg.lambda_e * losses["energy"]
        + cfg.lambda_g * losses["gate"]
        + cfg.lambda_h * losses["ent"]
    )
    return losses


def sme_score(logits):
    """Softmax energy (Klein et al. eq. 2, T=1): -log sum_k exp(softmax_k).

    NOTE: the notebooks apply an inner log (logsumexp(log(softmax))), which
    collapses to -log(sum softmax) = -log(1) = 0 for every sample. Without the
    inner log, ID samples get more negative scores than OOD samples.
    """
    probs = torch.softmax(logits, dim=-1)
    return -torch.logsumexp(probs, dim=-1)


@torch.no_grad()
def eval_dev(model, loader, cfg, device):
    model.eval()
    all_logits, all_y, all_ood, all_alpha = [], [], [], []
    for xh, xs, y, ood in loader:
        xh, xs = xh.to(device), xs.to(device)
        logits, alpha = model(xh, xs)
        all_logits.append(logits.cpu())
        all_y.append(y)
        all_ood.append(ood)
        all_alpha.append(alpha.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    ood = torch.cat(all_ood).bool()
    alpha = torch.cat(all_alpha)

    id_mask = (~ood) & (y >= 0)
    id_acc = 0.0
    if id_mask.any():
        id_acc = (logits[id_mask].argmax(-1) == y[id_mask]).float().mean().item()

    scores = sme_score(logits).numpy()
    id_scores = scores[id_mask.numpy()]
    ood_scores = scores[ood.numpy()]

    fpr95 = 1.0
    if len(id_scores) and len(ood_scores):
        thr = np.percentile(id_scores, 95)
        fpr95 = float((ood_scores <= thr).mean())

    alpha_id = alpha[id_mask].mean(0).tolist() if id_mask.any() else [0.5, 0.5]
    alpha_ood = alpha[ood].mean(0).tolist() if ood.any() else [0.5, 0.5]
    return {"id_acc": id_acc, "fpr95": fpr95, "alpha_id": alpha_id, "alpha_ood": alpha_ood}


def set_gate_trainable(model, trainable: bool):
    for p in model.gate.parameters():
        p.requires_grad = trainable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = Config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dual] device={device} epochs={cfg.epochs} batch={cfg.batch_size}")

    cores = load_npz(Path(cfg.cores_cache))
    ssl = load_npz(Path(cfg.xlsr_cache))
    data = join_caches(cores, ssl)
    print(f"[dual] joined {len(data['utt_ids'])} utterances "
          f"(cores={len(cores['utt_ids'])}, ssl={len(ssl['utt_ids'])})")

    def norm_stats(x, mask):
        mean = x[mask].mean(axis=0).astype(np.float32)
        std = x[mask].std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    train_mask = (data["splits"] == "train") & (~data["is_ood"]) & (data["label_ids"] >= 0)
    hc_mean, hc_std = norm_stats(data["x_hc"], train_mask)
    ssl_mean, ssl_std = norm_stats(data["x_ssl"], train_mask)
    x_hc = (data["x_hc"] - hc_mean) / hc_std
    x_ssl = (data["x_ssl"] - ssl_mean) / ssl_std
    print(f"[dual] normalized (train ID count {int(train_mask.sum())})")

    train_id_m = (data["splits"] == "train") & (~data["is_ood"]) & (data["label_ids"] >= 0)
    dev_ood_m = (data["splits"] == "dev") & (data["is_ood"])
    train_m = train_id_m | dev_ood_m
    dev_all_m = data["splits"] == "dev"

    train_loader = DataLoader(
        DualDataset(x_hc[train_m], x_ssl[train_m], data["label_ids"][train_m], data["is_ood"][train_m]),
        batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0)
    dev_loader = DataLoader(
        DualDataset(x_hc[dev_all_m], x_ssl[dev_all_m], data["label_ids"][dev_all_m], data["is_ood"][dev_all_m]),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"[dual] train rows {int(train_m.sum())} (ID {int(train_id_m.sum())} + Dev-OOD {int(dev_ood_m.sum())}) | "
          f"dev rows {int(dev_all_m.sum())}")

    model = DualBranchModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dual] model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    best_fpr95 = float("inf")
    history = []
    ckpt = Path(cfg.ckpt_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        gate_on = epoch > cfg.gate_freeze_epochs
        set_gate_trainable(model, gate_on)
        if epoch == cfg.gate_freeze_epochs + 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer.param_groups[0]["lr"],
                                          weight_decay=cfg.weight_decay)
            print(f"[dual] epoch {epoch}: gate UNFROZEN")

        model.train()
        running = {"total": 0.0, "ce": 0.0, "energy": 0.0, "gate": 0.0, "ent": 0.0}
        n_batches = 0
        for xh, xs, y, ood in train_loader:
            xh, xs, y, ood = xh.to(device), xs.to(device), y.to(device), ood.to(device)
            optimizer.zero_grad()
            logits, alpha = model(xh, xs)
            losses = total_loss(logits, alpha, y, ood, cfg)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for k in running:
                running[k] += float(losses[k].detach().cpu())
            n_batches += 1
        scheduler.step()
        for k in running:
            running[k] /= max(n_batches, 1)

        metrics = eval_dev(model, dev_loader, cfg, device)
        row = {"epoch": epoch, "gate_on": gate_on, **running, **metrics}
        history.append(row)

        if metrics["fpr95"] < best_fpr95:
            best_fpr95 = metrics["fpr95"]
            torch.save({
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "hc_mean": hc_mean, "hc_std": hc_std,
                "ssl_mean": ssl_mean, "ssl_std": ssl_std,
                "epoch": epoch,
                "metrics": metrics,
            }, ckpt)

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f"[epoch {epoch:03d}] gate={'ON' if gate_on else 'OFF'} | "
                  f"loss {running['total']:.3f} ce {running['ce']:.3f} | "
                  f"dev ID {metrics['id_acc']:.3f} FPR95 {metrics['fpr95']:.3f} | "
                  f"a_ssl ID/OOD {metrics['alpha_id'][1]:.3f}/{metrics['alpha_ood'][1]:.3f}")

    safe = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
            for r in history]
    with open(cfg.hist_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    print(f"[dual] done. best dev FPR95 = {best_fpr95:.3f} -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())