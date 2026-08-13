"""Local dual-branch evaluation -- port of Eval_OOD_Colab.ipynb with fixes.

  - sme_score fixed (notebook version is an identity, always 0).
  - EERc uses the MCE-2018 Top-1 definition (error at the threshold where
    ID-rejection/misclassification rate crosses OOD-acceptance rate), not the
    notebook's fixed-threshold approximation.
  - Dev thresholds are applied to Eval per scorer (paper Section 2.6).

Usage:
    python scripts/eval_dual_branch_local.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dual_branch_local import (  # noqa: E402
    Config, DualBranchModel, join_caches, load_npz,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_energy(logits):
    return (-torch.logsumexp(logits, dim=-1)).numpy()


def score_sme(logits):
    probs = torch.softmax(logits, dim=-1)
    return (-torch.logsumexp(probs, dim=-1)).numpy()


def score_msp(logits):
    probs = torch.softmax(logits, dim=-1)
    return (-probs.max(dim=-1).values).numpy()


def auroc(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos, n_neg = y_sorted.sum(), len(y_sorted) - y_sorted.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(y_sorted))
    ranks[order] = np.arange(1, len(y_sorted) + 1)
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eer(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    thresholds = np.unique(scores)
    best = 1.0
    for thr in thresholds:
        pred = scores > thr
        fpr = ((pred == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        fnr = ((pred == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        if abs(fpr - fnr) < best:
            best = abs(fpr - fnr)
            best_val = (fpr + fnr) / 2
    return float(best_val)


def eerc_mce(id_scores, id_preds, id_targets, ood_scores):
    """MCE-2018 Top-1: ID error = rejected (score>thr) OR misclassified;
    OOD error = accepted (score<=thr); EER at the crossing."""
    id_ok = (id_preds == id_targets).astype(bool)
    candidates = np.unique(np.concatenate([id_scores, ood_scores]))
    best = 1.0
    for thr in candidates:
        id_err = float((~(id_ok & (id_scores <= thr))).mean())
        ood_err = float((ood_scores <= thr).mean())
        if abs(id_err - ood_err) < best:
            best = abs(id_err - ood_err)
            best_val = 0.5 * (id_err + ood_err)
    return 100.0 * best_val


def fpr95(id_scores, ood_scores):
    thr = np.percentile(id_scores, 95)
    return float((ood_scores <= thr).mean()) * 100.0, float(thr)


@torch.no_grad()
def collect(data, x_hc, x_ssl, split, batch_size=256):
    m = data["splits"] == split
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_hc[m]).float(), torch.from_numpy(x_ssl[m]).float(),
                      torch.from_numpy(data["label_ids"][m]).long(),
                      torch.from_numpy(data["is_ood"][m].astype(bool))),
        batch_size=batch_size, shuffle=False)
    logits_l, y_l, ood_l = [], [], []
    for xh, xs, y, ood in loader:
        logits, _ = model(xh.to(DEVICE), xs.to(DEVICE))
        logits_l.append(logits.cpu())
        y_l.append(y)
        ood_l.append(ood)
    return {"logits": torch.cat(logits_l), "y": torch.cat(y_l), "ood": torch.cat(ood_l).bool()}


def evaluate_split(pack, thr=None):
    logits, y, ood = pack["logits"], pack["y"], pack["ood"]
    id_mask = (~ood) & (y >= 0)
    id_acc = (logits[id_mask].argmax(-1) == y[id_mask]).float().mean().item() if id_mask.any() else float("nan")
    results = {}
    for name, fn in [("Energy", score_energy), ("SME", score_sme), ("MSP", score_msp)]:
        scores = fn(logits)
        id_s = scores[id_mask.numpy()]
        ood_s = scores[ood.numpy()]
        if thr is None:
            fpr, t = fpr95(id_s, ood_s)
            thr_use = t
        else:
            thr_use = thr[name]
            fpr = float((ood_s <= thr_use).mean()) * 100.0
        results[name] = {
            "ID_Acc": id_acc,
            "FPR95": fpr,
            "EERc": eerc_mce(id_s, logits[id_mask].argmax(-1).numpy(), y[id_mask].numpy(), ood_s),
            "AUROC": auroc(id_s, ood_s),
            "OOD_EER": eer(id_s, ood_s),
            "thr": float(thr_use),
        }
    return results


if __name__ == "__main__":
    torch.set_num_threads(1)
    cfg = Config()
    ckpt_path = ROOT / "checkpoints" / "dual_branch_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = DualBranchModel(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')}, dev metrics {ckpt.get('metrics')})")

    cores = load_npz(ROOT / "cache" / "cores_features.npz")
    ssl = load_npz(ROOT / "cache" / "xlsr_features.npz")
    data = join_caches(cores, ssl)

    train_mask = (data["splits"] == "train") & (~data["is_ood"]) & (data["label_ids"] >= 0)
    hc_mean = data["x_hc"][train_mask].mean(0).astype(np.float32)
    hc_std = data["x_hc"][train_mask].std(0).astype(np.float32)
    hc_std[hc_std < 1e-6] = 1.0
    ssl_mean = data["x_ssl"][train_mask].mean(0).astype(np.float32)
    ssl_std = data["x_ssl"][train_mask].std(0).astype(np.float32)
    ssl_std[ssl_std < 1e-6] = 1.0
    x_hc = (data["x_hc"] - hc_mean) / hc_std
    x_ssl = (data["x_ssl"] - ssl_mean) / ssl_std

    dev = collect(data, x_hc, x_ssl, "dev")
    ev = collect(data, x_hc, x_ssl, "eval")

    dev_metrics = evaluate_split(dev)
    print("=== DEV ===")
    for k, v in dev_metrics.items():
        print(" ", k, v)

    thresholds = {name: dev_metrics[name]["thr"] for name in ["Energy", "SME", "MSP"]}
    eval_metrics = evaluate_split(ev, thr=thresholds)
    print("=== EVAL (thresholds from Dev) ===")
    for k, v in eval_metrics.items():
        print(" ", k, v)

    print("\npaper ref (Table 1, fused): ID acc 97.6% | SME AUROC 0.965 | FPR95 10.4 | EERc 4.98")