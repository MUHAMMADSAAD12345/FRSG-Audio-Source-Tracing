# FRSG-Audio-Source-Tracing

Re-implementation of **Dual-Branch Gated Fusion for Open-Set Audio Deepfake Source Tracing** (XLSR-53 + CORES) and follow-on improvements.

## Quick start (university GPU)

Follow **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — 5 steps:

1. Point notebooks at MLAAD + protocol CSVs  
2. Run `notebooks/CORES_Branch_Colab.ipynb` → `cores_features.npz`  
3. Run `notebooks/XLSR53_Extract_Colab.ipynb` (GPU) → `xlsr_features.npz`  
4. Run `notebooks/Train_Dual_Branch_Colab.ipynb` (GPU) → `dual_branch_best.pt`  
5. Run `notebooks/Eval_OOD_Colab.ipynb` → ID Acc / FPR95 / EERc  

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/CORES_Branch_Colab.ipynb`](notebooks/CORES_Branch_Colab.ipynb) | 66-d CORES extract + HC-only baseline |
| [`notebooks/XLSR53_Extract_Colab.ipynb`](notebooks/XLSR53_Extract_Colab.ipynb) | Frozen XLSR-53 → 1024-d cache |
| [`notebooks/Train_Dual_Branch_Colab.ipynb`](notebooks/Train_Dual_Branch_Colab.ipynb) | Gate + paper losses + training |
| [`notebooks/Eval_OOD_Colab.ipynb`](notebooks/Eval_OOD_Colab.ipynb) | SME / Energy / MSP OOD metrics |

## Home smoke test (no MLAAD)

```bash
python scripts/smoke_test_dual_branch.py
```
