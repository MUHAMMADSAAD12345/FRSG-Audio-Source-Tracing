# Uni GPU Runbook — Dual-Branch Gated Fusion

Do these steps **in order**. Do not redesign the architecture.

Paper: Dual-Branch Gated Fusion for Open-Set Audio Deepfake Source Tracing (XLSR-53 + CORES).

---

## Feature cache contract (both extractors must match)

Each cache is an `.npz` with aligned rows:

| Key | Type | Meaning |
|-----|------|---------|
| `utt_ids` | str array | Stable unique ID per utterance |
| `splits` | str array | `train` / `dev` / `eval` |
| `label_ids` | int64 | `0..23` for ID classes; `-1` for OOD |
| `is_ood` | bool | True if out-of-distribution system |
| `x_hc` | float32 `(N, 66)` | CORES only (`cores_features.npz`) |
| `x_ssl` | float32 `(N, 1024)` | XLSR-53 only (`xlsr_features.npz`) |

Training joins on `utt_id`. If IDs do not overlap, training will fail with a clear error.

Protocol CSV columns (when using real MLAAD):

```text
utt_id,wav_path,label_id,is_ood,split
```

---

## 5 steps at university

### 1. Paths

Put MLAAD audio + protocol CSVs somewhere accessible (Drive or local disk). Example:

```text
ROOT/
  data/
    MLAAD/          # wav files
    protocol/
      train.csv
      dev.csv
      eval.csv
  cache/
  checkpoints/
```

In every notebook, set:

```python
USE_DRIVE = True   # or False if data is local
ROOT = Path('/content/drive/MyDrive/mlaad-dual-branch')  # edit this
```

### 2. Extract CORES (CPU is fine)

Open [`notebooks/CORES_Branch_Colab.ipynb`](../notebooks/CORES_Branch_Colab.ipynb)

- Point `cfg.mlaad_root` / `cfg.protocol_dir` at real data
- Delete any old demo `cache/cores_features.npz`
- Run all cells → produces `cache/cores_features.npz`

### 3. Extract XLSR-53 (**GPU required**)

Open [`notebooks/XLSR53_Extract_Colab.ipynb`](../notebooks/XLSR53_Extract_Colab.ipynb)

- Same protocol paths as CORES
- Runtime → GPU
- Run all cells → produces `cache/xlsr_features.npz`
- Confirm printed overlap with CORES `utt_id`s is high / complete

### 4. Train dual-branch (**GPU**, ~150 epochs)

Open [`notebooks/Train_Dual_Branch_Colab.ipynb`](../notebooks/Train_Dual_Branch_Colab.ipynb)

- Confirm both caches exist
- Set `cfg.epochs = 150`, `cfg.batch_size = 128` for paper settings
- Run all cells → produces `checkpoints/dual_branch_best.pt`

Notes:
- Gate is **frozen for first 10 epochs** (paper)
- Checkpoint selection: lowest **Dev FPR95**
- Paper-faithful OOD aux uses **5× augmented Dev-OOD** (MUSAN + RIR). If you skip augmentation, training still runs but metrics will not match the paper.

### 5. Evaluate OOD

Open [`notebooks/Eval_OOD_Colab.ipynb`](../notebooks/Eval_OOD_Colab.ipynb)

- Loads best checkpoint + caches
- Reports ID Acc, FPR95, EERc, AUROC, OOD-EER (SME primary)
- Paste metrics back to the team chat / PR

Paper ballpark (Eval, SME): **~97.6% ID Acc, ~4.9% EERc, ~10.4% FPR95**

---

## What each person owns

| Person | Job |
|--------|-----|
| CORES owner | Keep CORES notebook correct; help debug 66-d cache |
| SSL owner | Run XLSR extract on GPU; verify `(N, 1024)` cache |
| Anyone with GPU | Run train + eval; report numbers |

---

## Home / no-GPU smoke test

`Train_Dual_Branch_Colab.ipynb` has **stub SSL mode**: if `xlsr_features.npz` is missing, it builds fake 1024-d vectors aligned to CORES `utt_id`s. That only checks the code path — **not** paper metrics.
