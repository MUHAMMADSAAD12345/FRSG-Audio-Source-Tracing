from __future__ import annotations
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

PIPE_ROOT = Path(r"D:\audio-ST\HC_Branch_1\full-dual-branch-pipeline")
SSL_MANIFEST = Path(r"D:\audio-ST\ssl_branch_1\ssl_branch\manifests\manifest.csv")
SSL_CACHE = Path(r"D:\audio-ST\ssl_branch_1\ssl_branch\cache\xlsr_embeddings")
AUDIO_ROOT = Path(r"D:\audio-ST\newww\audio-deepfake-source-tracing\data\MLAADv5")

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
N_MFCC = 13
N_CHROMA = 14


def extract_cores(path: str) -> np.ndarray:
    import librosa

    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)
    cepstral = np.vstack([mfcc, mfcc_d, mfcc_dd])
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP, n_chroma=N_CHROMA)
    zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=wav, frame_length=N_FFT, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)
    flatness = librosa.feature.spectral_flatness(y=wav, n_fft=N_FFT, hop_length=HOP)
    spectral = np.vstack([centroid, bandwidth, rolloff, contrast, flatness])
    T = min(f.shape[1] for f in [cepstral, chroma, zcr, rms, spectral])
    blocks = [cepstral[:, :T], chroma[:, :T], zcr[:, :T], rms[:, :T], spectral[:, :T]]
    frame_feats = np.vstack(blocks).T
    return frame_feats.mean(axis=0).astype(np.float32)


def main():
    rows = list(csv.DictReader(open(SSL_MANIFEST, encoding="utf-8")))
    print(f"manifest rows: {len(rows)}")

    proto_dir = PIPE_ROOT / "protocol_adapted"
    proto_dir.mkdir(exist_ok=True)
    for split in ("train", "dev", "eval"):
        out = proto_dir / f"{split}.csv"
        if out.exists():
            print(f"skip {out} (exists)")
            continue
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "wav_path", "label_id", "is_ood", "split"])
            for r in rows:
                if r["split"] != split:
                    continue
                is_ood = r["is_ood"].lower() in ("1", "true", "yes")
                w.writerow([
                    r["utt_id"],
                    str(AUDIO_ROOT / r["filepath"]),
                    -1 if is_ood else int(r["class_idx"]),
                    is_ood,
                    split,
                ])
        print(f"wrote {out}")

    utt_ids, splits, label_ids, is_oods = [], [], [], []
    for r in rows:
        utt_ids.append(r["utt_id"])
        splits.append(r["split"])
        is_ood = r["is_ood"].lower() in ("1", "true", "yes")
        is_oods.append(is_ood)
        label_ids.append(-1 if is_ood else int(r["class_idx"]))
    utt_ids = np.array(utt_ids)
    splits = np.array(splits)
    label_ids = np.array(label_ids, dtype=np.int64)
    is_oods = np.array(is_oods, dtype=bool)

    xlsr_out = PIPE_ROOT / "cache" / "xlsr_features.npz"
    if not xlsr_out.exists():
        x_ssl = np.stack([np.load(SSL_CACHE / f"{u}.npy") for u in utt_ids]).astype(np.float32)
        print("x_ssl", x_ssl.shape)
        np.savez_compressed(xlsr_out, utt_ids=utt_ids, splits=splits,
                            label_ids=label_ids, is_ood=is_oods, x_ssl=x_ssl)
    else:
        print(f"skip {xlsr_out} (exists)")

    cores_out = PIPE_ROOT / "cache" / "cores_features.npz"
    if cores_out.exists():
        print(f"skip {cores_out} (exists)")
        return
    paths = [str(AUDIO_ROOT / r["filepath"]) for r in rows]
    n_workers = os.cpu_count() or 8
    with Pool(n_workers) as pool:
        feats = pool.map(extract_cores, paths, chunksize=16)
    x_hc = np.stack(feats).astype(np.float32)
    print("x_hc", x_hc.shape)
    np.savez_compressed(cores_out, utt_ids=utt_ids, splits=splits,
                        label_ids=label_ids, is_ood=is_oods, x_hc=x_hc)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
