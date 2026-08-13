from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
EPOCHS = 150
GATE_FREEZE_EPOCHS = 10
SEED = 42


class Config:
    cores_dim = 66
    ssl_dim = 1024
    expert_hidden = 512
    expert_out = 256
    gate_hidden = 128
    dropout_expert = 0.3
    dropout_gate = 0.2
    num_classes = 24
    lr = 1e-4
    weight_decay = 1e-4
    label_smoothing = 0.15
    grad_clip = 5.0
    min_lr = 5e-6
    lambda_e = 0.5
    lambda_g = 0.05
    lambda_h = 0.3
    m_in = -15.0
    m_out = -2.0
    cores_cache = str(ROOT / "cache" / "cores_features.npz")
    xlsr_cache = str(ROOT / "cache" / "xlsr_features.npz")
    ckpt_path = str(ROOT / "checkpoints" / "dual_branch_best.pt")
    hist_path = str(ROOT / "checkpoints" / "train_history.json")


cfg = Config()


def load_npz(path):
    loaded = np.load(path, allow_pickle=True)
    return {k: loaded[k] for k in loaded.files}


def join_caches(cores, ssl):
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


class ExpertMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, in_dim=512, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

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
        logits = self.classifier(e_fused)
        return logits, alpha, e_hc, e_ssl, e_fused


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


def total_loss(logits, alpha, y, is_ood, cfg: Config):
    id_mask = (~is_ood) & (y >= 0)
    if id_mask.any():
        ce = label_smoothed_ce(logits[id_mask], y[id_mask], cfg.num_classes, cfg.label_smoothing)
    else:
        ce = logits.new_tensor(0.0)
    energy = energy_margin_loss(logits, is_ood, cfg.m_in, cfg.m_out)
    gate = gate_diversity_loss(alpha, is_ood)
    ent = gate_entropy_loss(alpha)
    total = ce + cfg.lambda_e * energy + cfg.lambda_g * gate + cfg.lambda_h * ent
    return {"total": total, "ce": ce, "energy": energy, "gate": gate, "ent": ent}


def sme_score(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.logsumexp(torch.log(probs.clamp_min(1e-12)), dim=-1)


@torch.no_grad()
def eval_dev(model, loader, cfg: Config):
    model.eval()
    all_logits, all_y, all_ood = [], [], []
    for xh, xs, y, ood in loader:
        xh, xs = xh.to(DEVICE), xs.to(DEVICE)
        logits, *_ = model(xh, xs)
        all_logits.append(logits.cpu())
        all_y.append(y)
        all_ood.append(ood)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    ood = torch.cat(all_ood).bool()
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
    return {"id_acc": id_acc, "fpr95": fpr95}


def set_gate_trainable(model, trainable):
    for p in model.gate.parameters():
        p.requires_grad = trainable


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"device = {DEVICE}")

    cores = load_npz(Path(cfg.cores_cache))
    ssl = load_npz(Path(cfg.xlsr_cache))
    data = join_caches(cores, ssl)
    print(f"joined: {len(data['utt_ids'])} (cores={len(cores['utt_ids'])}, ssl={len(ssl['utt_ids'])})")

    train_mask = (data["splits"] == "train") & (~data["is_ood"]) & (data["label_ids"] >= 0)
    hc_mean = data["x_hc"][train_mask].mean(axis=0).astype(np.float32)
    hc_std = data["x_hc"][train_mask].std(axis=0).astype(np.float32)
    ssl_mean = data["x_ssl"][train_mask].mean(axis=0).astype(np.float32)
    ssl_std = data["x_ssl"][train_mask].std(axis=0).astype(np.float32)
    hc_std[hc_std < 1e-6] = 1.0
    ssl_std[ssl_std < 1e-6] = 1.0
    x_hc = (data["x_hc"] - hc_mean) / hc_std
    x_ssl = (data["x_ssl"] - ssl_mean) / ssl_std
    print(f"train ID count: {int(train_mask.sum())}")

    train_id_m = train_mask
    dev_ood_m = (data["splits"] == "dev") & (data["is_ood"])
    train_m = train_id_m | dev_ood_m
    dev_all_m = data["splits"] == "dev"
    print(f"train rows: {int(train_m.sum())} (ID {int(train_id_m.sum())} + dev-OOD {int(dev_ood_m.sum())})")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_hc[train_m]).float(),
            torch.from_numpy(x_ssl[train_m]).float(),
            torch.from_numpy(data["label_ids"][train_m]).long(),
            torch.from_numpy(data["is_ood"][train_m].astype(np.bool_)),
        ),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True,
    )
    dev_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_hc[dev_all_m]).float(),
            torch.from_numpy(x_ssl[dev_all_m]).float(),
            torch.from_numpy(data["label_ids"][dev_all_m]).long(),
            torch.from_numpy(data["is_ood"][dev_all_m].astype(np.bool_)),
        ),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
    )

    model = DualBranchModel(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"dual-branch params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=cfg.min_lr)

    ckpt_dir = Path(cfg.ckpt_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_fpr95 = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        gate_on = epoch > GATE_FREEZE_EPOCHS
        set_gate_trainable(model, gate_on)
        if epoch == GATE_FREEZE_EPOCHS + 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer.param_groups[0]["lr"],
                                          weight_decay=cfg.weight_decay)
            print(f"epoch {epoch}: gate UNFROZEN")

        model.train()
        running = {"total": 0.0, "ce": 0.0, "energy": 0.0, "gate": 0.0, "ent": 0.0}
        n_batches = 0
        for xh, xs, y, ood in train_loader:
            xh, xs, y, ood = xh.to(DEVICE), xs.to(DEVICE), y.to(DEVICE), ood.to(DEVICE)
            optimizer.zero_grad()
            logits, alpha, *_ = model(xh, xs)
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

        metrics = eval_dev(model, dev_loader, cfg)
        history.append({"epoch": epoch, "gate_on": gate_on, **running, **metrics})

        if metrics["fpr95"] < best_fpr95:
            best_fpr95 = metrics["fpr95"]
            torch.save({
                "model": model.state_dict(),
                "cfg": {k: getattr(cfg, k) for k in (
                    "cores_dim", "ssl_dim", "expert_hidden", "expert_out", "gate_hidden",
                    "dropout_expert", "dropout_gate", "num_classes")},
                "hc_mean": hc_mean, "hc_std": hc_std,
                "ssl_mean": ssl_mean, "ssl_std": ssl_std,
                "epoch": epoch,
                "metrics": metrics,
            }, cfg.ckpt_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:03d} gate={'ON' if gate_on else 'OFF'} | "
                  f"loss {running['total']:.3f} ce {running['ce']:.3f} | "
                  f"dev ID {metrics['id_acc']:.3f} FPR95 {metrics['fpr95']:.3f}")

    safe = [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in r.items()} for r in history]
    with open(cfg.hist_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    print(f"best dev FPR95: {best_fpr95:.3f}")
    print(f"checkpoint: {cfg.ckpt_path}")
    print(f"history: {cfg.hist_path}")


if __name__ == "__main__":
    sys.exit(main())
