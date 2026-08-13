# SSL Branch Implementation — Handoff Summary

## Project context

Reproducing: **"Dual-Branch Gated Fusion for Open-Set Audio Deepfake Source Tracing"** (Khan et al., 2026), specifically the **SSL branch only**. A teammate is separately implementing the CORES (handcrafted, 66-d) branch. The two branches will be merged later into the paper's gated fusion architecture — that merge/fusion stage has **not** been built yet and is out of scope for this codebase.

Paper's architecture (for reference):
```
raw audio → frozen XLSR-53 → mean-pool → x_ssl ∈ R^1024
x_ssl → projection network (FC 1024→512, BatchNorm, ReLU, Dropout 0.3, FC 512→256) → e_ssl ∈ R^256
[e_hc; e_ssl] ∈ R^512 → gating network (2-layer, 512→128→2, softmax) → [α_hc, α_ssl]
e_fused = α_hc·e_hc + α_ssl·e_ssl → linear classifier → 24 ID classes
```
Losses: label-smoothed CE (ε=0.15) + energy margin loss (Liu et al., m_in=-15.0, m_out=-2.0, λ=0.5) + gate diversity loss (KL divergence, λ=0.05) + gate entropy (λ=0.3). AdamW lr=1e-4, wd=1e-4, cosine annealing to 5e-6, batch 128, grad clip 5.0, 150 epochs.

Paper's target numbers (Table 3, full fusion system): **97.6% ID accuracy, 4.9% EERc, 10.4% FPR95** (SME scorer). SSL-only ablation (Figure 2, for comparison): **96.1% ID accuracy, 96.2% FPR95** (i.e. good at classification, bad at OOD rejection alone — this is why fusion is needed).

## What has been built: `ssl_branch/` project

Full project scaffold (Python, PyTorch + HuggingFace transformers), structured as:
```
ssl_branch/
├── configs/config.yaml          # all hyperparameters + paths, matches paper values
├── src/
│   ├── data/
│   │   ├── build_manifest.py    # builds manifests/manifest.csv from audio + protocol
│   │   └── dataset.py           # CachedEmbeddingDataset — reads cached .npy embeddings
│   ├── features/
│   │   └── extract_xlsr.py      # frozen XLSR-53 → mean-pooled 1024-d embeddings, cached to disk
│   ├── models/
│   │   └── ssl_branch.py        # SSLProjection (1024→512→256, the fusion hand-off piece)
│   │                             # + SSLBranchClassifier (projection + linear head, for standalone training)
│   ├── losses/
│   │   └── losses.py            # label_smoothed_ce, energy_margin_loss (used)
│   │                             # + gate_diversity_loss, gate_entropy_loss (written for later fusion stage, NOT used yet)
│   ├── utils/
│   │   ├── config.py, seed.py, metrics.py  # metrics.py has FPR95/EER/AUROC implementations
│   ├── train.py                 # trains SSLBranchClassifier on cached embeddings
│   └── evaluate.py              # ID accuracy + Energy/MSP scorer OOD metrics (AUROC/FPR95/EER)
└── scripts/*.sh                 # convenience runners
```

Key design decisions:
- **Cached-embedding architecture**: `extract_xlsr.py` runs once, caches 1024-d `.npy` per utterance keyed by a UUID `utt_id` (assigned when the manifest is built). All downstream code (train/eval) reads only from this cache, never touches raw audio again. This makes extraction resumable (skips already-cached files) and portable across machines — **but the manifest.csv must travel with the cache**, since utt_ids are the join key. Regenerating the manifest orphans any existing cache.
- **`SSLProjection` is the deliverable for fusion hand-off** — it's a standalone `nn.Module` (1024→512→256) that whoever builds the joint fusion model can import directly and load trained weights into.
- **`SSLBranchClassifier`** wraps `SSLProjection` + a linear classifier so the branch can be trained/evaluated standalone (reproduces the paper's SSL-only ablation as a sanity baseline).
- Manifest schema: `utt_id, filepath, language, model_name, is_ood (bool), class_idx (int, -1 for OOD), split (train/dev/eval)`.

## What has been done (chronological)

1. Built the full project scaffold above.
2. **First run used an "auto-split"** (`build_manifest.py`'s fallback mode: infers class = folder name, randomly holds out a fraction of *systems* as OOD, since we didn't yet have the official paper protocol). Ran on a personal ~11,877-file MLAAD-style dataset (`fake/<lang>/<model_name>/*.wav`, e.g. `tts_models_multilingual_multi-dataset_xtts_v2`) → produced only **7 ID / 3 OOD systems** (too small/easy, not comparable to the paper).
3. Debugged environment issues along the way (Windows-specific):
   - `torchaudio.load()` failing due to missing `torchcodec`/FFmpeg backend on Windows — **fix applied**: swapped `WavDataset.__getitem__` in `extract_xlsr.py` to load via `soundfile.read()` instead of `torchaudio.load()` (torchaudio's `resample` still used afterward, only the *loading* call changed).
4. Ran extraction on CPU (laptop, no GPU) — extremely slow (~2%/hour, would've taken ~50h). Moved to Google Colab (GPU) instead: zipped `ssl_branch/` (code+manifest+partial cache) separately from the audio (uploaded to Drive), remapped Windows file paths in `manifest.csv` to Colab paths via a one-off pandas script, reran extraction on GPU — completed successfully.
5. Trained + evaluated on this **auto-split (7 ID / 3 OOD)** data as a first pipeline sanity check. Result:
   ```
   ID_Accuracy_%: 99.58
   Energy_AUROC: 0.8774 | Energy_FPR95_%: 50.33 | Energy_EER_%: 22.11
   MSP_AUROC: 0.9681    | MSP_FPR95_%: 20.67    | MSP_EER_%: 8.31
   ```
   Confirmed the pipeline is wired correctly end-to-end, but these numbers are **not comparable to the paper** — the 7/3 split is much easier than the paper's 24 ID / 43 OOD benchmark, so FPR95 looks artificially good.
6. **Located and downloaded the official paper protocol**: cloned `https://github.com/piotrkawa/audio-deepfake-source-tracing` (the official Interspeech 2025 Source Tracing Special Session baseline repo), ran `python scripts/download_resources.py`. This pulled:
   - `data/MLAADv5/` — the actual multi-part-zipped MLAAD v5 audio archive (`mlaad_v5.z01`–`z10` + `.zip`)
   - `data/MLAADv5_for_sourcetracing/mlaadv5_for_sourcetracing/` — the **official protocol CSVs**: `train.csv`, `dev.csv`, `eval.csv`, plus fine-grained OOD breakdown CSVs (`dev___lang_not_seen___model_not_seen.csv` etc.) and `meta.txt`.
   - Also had to work around several Windows pip install issues with this baseline repo's `requirements.txt` (unrelated to our own `ssl_branch` project): commented out Linux/CUDA-only pinned packages (`nvidia-*-cu12`, `triton==3.1.0`), fixed a `torch==2.5.1` vs `torchaudio==2.5.0` version mismatch (pinned both to `2.5.0`), and resolved a Windows permission error by installing into a venv (`python -m venv .venv`).
7. Extracted the multi-part archives and **rebuilt `manifests/manifest.csv` using the real MLAAD v5 audio + official protocol CSVs** (via `build_manifest.py`'s `protocol_csv` mode) instead of the auto-split. This is confirmed done — manifest file exists.

## Current state / what's NOT yet done

- **Have not yet verified the class counts** in the new protocol-based manifest match the paper's Table 2 (Train: 24 ID systems / 11,000 samples; Dev: 8 ID + 17 OOD systems / 4,800 + 7,200 samples; Eval: 21 ID + 43 OOD systems / 13,591 + 20,309 samples). Should run:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('manifests/manifest.csv'); print(df['split'].value_counts()); print('ID classes:', df.loc[~df['is_ood'],'class_idx'].nunique()); print('OOD systems:', df.loc[df['is_ood'],'model_name'].nunique())"
  ```
- **Have NOT re-run feature extraction** on this new, larger, official dataset. The existing `cache/xlsr_embeddings/*.npy` files are from the OLD 11,877-file auto-split dataset and are very likely a different (probably smaller) file set than the official MLAAD v5 protocol — cache will need to be regenerated (or substantially extended) for the new manifest's utt_ids. This should be done on GPU (Colab), not CPU.
- **Have NOT retrained** `SSLBranchClassifier` on the official split.
- **Have NOT re-evaluated** on the official split.
- **Fusion stage (gate, combining with teammate's CORES branch) has not been started** — `gate_diversity_loss` and `gate_entropy_loss` exist in `src/losses/losses.py` but are unused placeholders for whoever assembles the joint model.
- Should double check whether `paths.audio_root` in `configs/config.yaml` needs updating to point at the extracted `MLAADv5` folder, and whether `features.max_audio_seconds` (currently 6.0s) is appropriate for this dataset's utterance lengths.

## Immediate next steps (in order)

1. Verify manifest class counts against paper's Table 2 (command above).
2. Re-run `python -m src.features.extract_xlsr --config configs/config.yaml` on GPU against the new manifest/official audio.
3. Re-run `python -m src.train --config configs/config.yaml` — confirm log line shows ~24 ID classes.
4. Re-run `python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/ssl_branch_best.pt` — compare against paper's SSL-only ablation (96.1% ID acc / 96.2% FPR95) as the sanity check, since this standalone branch is expected to reproduce roughly that failure mode (good ID acc, poor OOD rejection) prior to fusion.
5. Once satisfied with standalone SSL branch numbers, prepare hand-off artifacts for the teammate merging branches: trained `SSLProjection` weights (already saved inside `checkpoints/ssl_branch_best.pt` under key `projection_state`) + cached embeddings.

## Known environment notes for whoever continues this

- User is on Windows, GPU (NVIDIA RTX 4000 Ada Gene with 20GB VRAM)
- When moving between machines: **always carry `manifest.csv` together with `cache/xlsr_embeddings/`** — utt_ids are the join key; regenerating the manifest breaks the cache mapping.
- `torchaudio.load()` is broken on this Windows setup without extra FFmpeg/torchcodec setup — `extract_xlsr.py` already patched to use `soundfile.read()` instead; do not revert this.
- The separate baseline repo (`piotrkawa/audio-deepfake-source-tracing`) is only being used as a **data/protocol source** (via its `download_resources.py`), not as the modeling codebase — our own `ssl_branch/` project is the actual implementation being built and should remain the base going forward.
