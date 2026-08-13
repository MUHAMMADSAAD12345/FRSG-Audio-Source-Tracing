from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512

CKPT = ROOT / "checkpoints" / "dual_branch_best.pt"
CORES_CACHE = ROOT / "cache" / "cores_features.npz"
XLSR_CACHE = ROOT / "cache" / "xlsr_features.npz"


class ExpertMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=512, out=256, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(inplace=True), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out), torch.nn.BatchNorm1d(out), torch.nn.ReLU(inplace=True), torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(torch.nn.Module):
    def __init__(self, in_dim=512, hidden=128, dropout=0.2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(inplace=True), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 2))

    def forward(self, e_hc, e_ssl):
        return torch.softmax(self.net(torch.cat([e_hc, e_ssl], dim=-1)), dim=-1)


class DualBranchModel(torch.nn.Module):
    def __init__(self, cores_dim=66, ssl_dim=1024, hidden=512, out=256,
                 gate_hidden=128, num_classes=24, drop_e=0.3, drop_g=0.2):
        super().__init__()
        self.expert_hc = ExpertMLP(cores_dim, hidden, out, drop_e)
        self.expert_ssl = ExpertMLP(ssl_dim, hidden, out, drop_e)
        self.gate = GatingNetwork(out * 2, gate_hidden, drop_g)
        self.classifier = torch.nn.Linear(out, num_classes)

    def forward(self, x_hc, x_ssl):
        e_hc = self.expert_hc(x_hc)
        e_ssl = self.expert_ssl(x_ssl)
        alpha = self.gate(e_hc, e_ssl)
        e_fused = alpha[:, 0:1] * e_hc + alpha[:, 1:2] * e_ssl
        return self.classifier(e_fused), alpha


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def join_caches(cores, ssl):
    ssl_map = {uid: i for i, uid in enumerate(ssl["utt_ids"].tolist())}
    ic, is_ = [], []
    for i, uid in enumerate(cores["utt_ids"].tolist()):
        if uid in ssl_map:
            ic.append(i)
            is_.append(ssl_map[uid])
    ic, is_ = np.asarray(ic), np.asarray(is_)
    return {
        "utt_ids": cores["utt_ids"][ic],
        "splits": cores["splits"][ic],
        "label_ids": cores["label_ids"][ic],
        "is_ood": cores["is_ood"][ic],
        "x_hc": cores["x_hc"][ic].astype(np.float32),
        "x_ssl": ssl["x_ssl"][is_].astype(np.float32),
    }


def score_energy(logits):
    return (-torch.logsumexp(logits, dim=-1)).numpy()


def score_sme(logits):
    probs = torch.softmax(logits, dim=-1)
    return (-torch.logsumexp(torch.log(probs.clamp_min(1e-12)), dim=-1)).numpy()


def score_msp(logits):
    probs = torch.softmax(logits, dim=-1)
    return (-probs.max(dim=-1).values).numpy()


def fpr95(id_scores, ood_scores):
    thr = np.percentile(id_scores, 95)
    return float((ood_scores <= thr).mean()) * 100.0, float(thr)


def auroc(id_scores, ood_scores):
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))


def ood_eer(id_scores, ood_scores):
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0) * 100.0


def eerc_mce2018(id_scores, ood_scores, id_preds, id_targets):
    candidates = np.sort(np.concatenate([id_scores, ood_scores]))
    best_diff = np.inf
    for theta in candidates:
        id_err = ((id_scores > theta) | (id_preds != id_targets)).mean()
        ood_err = (ood_scores <= theta).mean()
        diff = abs(id_err - ood_err)
        if diff < best_diff:
            best_diff = diff
            best_err = (id_err + ood_err) / 2.0
    return float(best_err) * 100.0


@torch.no_grad()
def collect(x_hc, x_ssl, label_ids, is_ood):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_hc).float(),
            torch.from_numpy(x_ssl).float(),
            torch.from_numpy(label_ids).long(),
            torch.from_numpy(is_ood.astype(np.bool_)),
        ),
        batch_size=BATCH_SIZE, shuffle=False)
    logits_l, y_l, ood_l, a_l = [], [], [], []
    for xh_b, xs_b, y, ood in loader:
        logits, alpha = model(xh_b.to(DEVICE), xs_b.to(DEVICE))
        logits_l.append(logits.cpu())
        y_l.append(y)
        ood_l.append(ood)
        a_l.append(alpha.cpu())
    return {
        "logits": torch.cat(logits_l),
        "y": torch.cat(y_l),
        "ood": torch.cat(ood_l).bool(),
        "alpha": torch.cat(a_l),
    }


def evaluate_split(pack, thr=None):
    logits, y, ood = pack["logits"], pack["y"], pack["ood"]
    id_mask = (~ood) & (y >= 0)
    id_acc = (logits[id_mask].argmax(-1) == y[id_mask]).float().mean().item() if id_mask.any() else float("nan")
    results = {}
    for name, fn in [("Energy", score_energy), ("SME", score_sme), ("MSP", score_msp)]:
        scores = fn(logits)
        id_s = scores[id_mask.numpy()]
        ood_s = scores[ood.numpy()]
        fpr, t = fpr95(id_s, ood_s) if thr is None else (float((ood_s <= thr[name]).mean()) * 100.0, thr[name])
        preds = logits[id_mask].argmax(-1).numpy()
        y_id = y[id_mask].numpy()
        results[name] = {
            "ID_Acc": id_acc * 100.0,
            "FPR95": fpr,
            "EERc": eerc_mce2018(id_s, ood_s, preds, y_id),
            "AUROC": auroc(id_s, ood_s),
            "OOD_EER": ood_eer(id_s, ood_s),
            "thr": float(t),
        }
    alpha_id = pack["alpha"][id_mask].mean(0).tolist() if id_mask.any() else [None, None]
    alpha_ood = pack["alpha"][ood].mean(0).tolist() if ood.any() else [None, None]
    results["gate"] = {"alpha_id": alpha_id, "alpha_ood": alpha_ood}
    return results


def main():
    print(f"device = {DEVICE}")
    ckpt = torch.load(CKPT, map_location=DEVICE)
    c = ckpt["cfg"]
    global model
    model = DualBranchModel(
        cores_dim=c.get("cores_dim", 66),
        ssl_dim=c.get("ssl_dim", 1024),
        hidden=c.get("expert_hidden", 512),
        out=c.get("expert_out", 256),
        gate_hidden=c.get("gate_hidden", 128),
        num_classes=c.get("num_classes", 24),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"ckpt epoch {ckpt.get('epoch')} metrics {ckpt.get('metrics')}")

    cores = load_npz(CORES_CACHE)
    ssl = load_npz(XLSR_CACHE)
    data = join_caches(cores, ssl)
    hc_mean, hc_std = ckpt["hc_mean"], ckpt["hc_std"]
    ssl_mean, ssl_std = ckpt["ssl_mean"], ckpt["ssl_std"]
    x_hc = (data["x_hc"] - hc_mean) / hc_std
    x_ssl = (data["x_ssl"] - ssl_mean) / ssl_std
    print(f"joined {len(data['utt_ids'])} utterances")

    packs = {}
    for split in ("dev", "eval"):
        m = data["splits"] == split
        packs[split] = collect(x_hc[m], x_ssl[m], data["label_ids"][m], data["is_ood"][m])
        print(f"{split}: {len(packs[split]['y'])}")

    dev_metrics = evaluate_split(packs["dev"])
    print("=== DEV ===")
    print(json.dumps(dev_metrics, indent=2))
    thresholds = {name: dev_metrics[name]["thr"] for name in ["Energy", "SME", "MSP"]}
    eval_metrics = evaluate_split(packs["eval"], thr=thresholds)
    print("=== EVAL (thresholds from Dev) ===")
    print(json.dumps(eval_metrics, indent=2))

    out = {"dev": dev_metrics, "eval": eval_metrics}
    with open(ROOT / "checkpoints" / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", ROOT / "checkpoints" / "eval_results.json")


if __name__ == "__main__":
    sys.exit(main())
