"""Build protocol CSVs in the dual-branch pipeline's contract format.

The official MLAAD source-tracing protocol CSVs (train/dev/eval.csv) contain
only `path,model_name`. The notebooks expect `utt_id,wav_path,label_id,
is_ood,split` (see docs/RUNBOOK.md). This script bridges the two:

  - utt_id: sha1(protocol-relative path)[:12] -- the SAME scheme used by the
    SSL branch cache, so the XLSR features join cleanly at training time.
  - label_id: 0..23 for the 24 in-domain (train.csv) models, alphabetical;
    -1 for out-of-domain models.
  - is_ood: True iff model_name not in train.csv (paper Table 2 rule).
  - wav_path: absolute path under audio_root.

Outputs: <out_dir>/train.csv, dev.csv, eval.csv (one row per protocol row).

Usage:
    python scripts/build_protocol_csvs.py \
        --protocol-dir <raw protocol dir> \
        --audio-root <MLAAD wav root> \
        --out-dir data/protocol
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

PROTOCOL_FILES = ("train.csv", "dev.csv", "eval.csv")


def utt_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    frames = []
    for fname in PROTOCOL_FILES:
        p = args.protocol_dir / fname
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if set(rows[0].keys()) != {"path", "model_name"}:
            raise ValueError(f"{p} has columns {list(rows[0].keys())}; expected ['path','model_name']")
        for r in rows:
            r["split"] = fname[:-4]
        frames.append(rows)
        print(f"{fname}: {len(rows)} rows")

    proto = [r for fr in frames for r in fr]
    id_models = sorted({r["model_name"] for r in proto if r["split"] == "train"})
    id_to_idx = {m: i for i, m in enumerate(id_models)}
    print(f"ID models (train.csv): {len(id_models)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for split in ("train", "dev", "eval"):
        out = args.out_dir / f"{split}.csv"
        n_ood = 0
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "wav_path", "label_id", "is_ood", "split"])
            for r in proto:
                if r["split"] != split:
                    continue
                is_ood = r["model_name"] not in id_to_idx
                n_ood += is_ood
                wav_path = (args.audio_root / r["path"].lstrip("./")).resolve()
                if not wav_path.exists():
                    missing += 1
                w.writerow([utt_id(r["path"]), str(wav_path),
                            id_to_idx.get(r["model_name"], -1), is_ood, split])
        print(f"{split}.csv: ID={sum(1 for r in proto if r['split']==split and not (r['model_name'] not in id_to_idx))} "
              f"OOD={n_ood} -> {out}")
    if missing:
        print(f"WARNING: {missing} wav paths missing on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())