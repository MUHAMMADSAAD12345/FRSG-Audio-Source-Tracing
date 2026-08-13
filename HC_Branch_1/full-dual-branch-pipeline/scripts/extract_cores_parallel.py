"""Parallel CORES feature extraction (66-d) for the dual-branch pipeline.

Implements the exact CORES descriptor from the CORES_Branch_Colab notebook
(39 MFCC+delta+delta-delta, 14 chroma, 1 ZCR, 1 RMS, 11 spectral
[centroid, bandwidth, rolloff, contrast x7, flatness]) as a frame-level
extractor mean-pooled to one 66-d vector per utterance. Same contract as the
notebook's cores_features.npz:

    utt_ids    (N,)      str
    splits     (N,)      str   train / dev / eval
    label_ids  (N,)      int64 0..23 for ID classes; -1 for OOD
    is_ood     (N,)      bool
    x_hc       (N, 66)   float32

Per-utterance vectors are written to <out>/cache/cores_tmp/{utt_id}.npy by a
process pool (resumable: existing files are skipped), then assembled into the
final npz. Run with --force to recompute everything.

Usage:
    python scripts/extract_cores_parallel.py \
        --protocol-dir data/protocol --audio-root <MLAAD wav root> \
        --out <pipeline root> [--workers N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import soundfile as sf

import librosa

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
CORES_DIM = 66


def extract_cores(wav: np.ndarray, sr: int) -> np.ndarray:
    """Frame-level CORES features -> mean-pooled 66-d utterance vector."""
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)
    cepstral = np.vstack([mfcc, mfcc_d, mfcc_dd])  # (39, T)

    chroma = librosa.feature.chroma_stft(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP, n_chroma=14)
    zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=wav, frame_length=N_FFT, hop_length=HOP)

    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    flatness = librosa.feature.spectral_flatness(y=wav, n_fft=N_FFT, hop_length=HOP)
    spectral = np.vstack([centroid, bandwidth, rolloff, contrast, flatness])  # (11, T)

    T = min(f.shape[1] for f in [cepstral, chroma, zcr, rms, spectral])
    blocks = [cepstral[:, :T], chroma[:, :T], zcr[:, :T], rms[:, :T], spectral[:, :T]]
    frame_feats = np.vstack(blocks).T  # (T, 66)
    utterance = frame_feats.mean(axis=0)
    assert utterance.shape[0] == CORES_DIM
    return utterance.astype(np.float32)


def worker(job: tuple) -> tuple[str, int]:
    """job = (utt_id, wav_path, out_path). Returns (utt_id, bytes or 0)."""
    utt_id, wav_path, out_path = job
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    feat = extract_cores(wav.astype(np.float32), sr)
    np.save(out_path, feat)
    return utt_id, feat.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True, type=Path, help="dir with the contract CSVs")
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="pipeline root (cache/cores_*.npz written here)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # librosa's internal workers (numba/scipy/OpenMP) must stay single-threaded
    # per process, otherwise 10+ workers exhaust RAM on Windows.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    if args.workers is None:
        args.workers = max(1, min(os.cpu_count() - 2, 8))
    print(f"workers: {args.workers}")

    rows = []
    for split in ("train", "dev", "eval"):
        p = args.protocol_dir / f"{split}.csv"
        import csv
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["utt_id"], r["wav_path"], r["split"],
                             int(r["label_id"]), r["is_ood"].lower() in ("1", "true", "yes")))

    tmp_dir = args.out / "cache" / "cores_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs, n_skip = [], 0
    for utt_id, wav_path, _split, _lab, _ood in rows:
        out_path = tmp_dir / f"{utt_id}.npy"
        if out_path.exists() and not args.force:
            n_skip += 1
            continue
        jobs.append((utt_id, wav_path, str(out_path)))
    print(f"total rows: {len(rows)} | to extract: {len(jobs)} | already cached: {n_skip}")

    if jobs:
        with mp.Pool(args.workers) as pool:
            done = 0
            for _utt, _n in pool.imap_unordered(worker, jobs, chunksize=16):
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(jobs)} utterances extracted", flush=True)
            pool.close()
            pool.join()
        print(f"extraction done: {len(jobs)} utterances")

    utt_ids, splits, label_ids, is_ood, x_hc = [], [], [], [], []
    missing = 0
    for utt_id, wav_path, split, label_id, ood in rows:
        p = tmp_dir / f"{utt_id}.npy"
        if not p.exists():
            missing += 1
            continue
        utt_ids.append(utt_id)
        splits.append(split)
        label_ids.append(label_id)
        is_ood.append(ood)
        x_hc.append(np.load(p))
    if missing:
        print(f"WARNING: {missing} utterances missing after extraction")
    x_hc = np.stack(x_hc).astype(np.float32)
    np.savez_compressed(
        args.out / "cache" / "cores_features.npz",
        utt_ids=np.array(utt_ids),
        splits=np.array(splits),
        label_ids=np.array(label_ids, dtype=np.int64),
        is_ood=np.array(is_ood, dtype=bool),
        x_hc=x_hc,
    )
    print(f"wrote {args.out / 'cache' / 'cores_features.npz'}  x_hc shape {x_hc.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())