# EfficientAD Training & PUAD Inference

## Overview
This workspace contains a three-stage pipeline for EfficientAD anomaly detection:
LLMs were used for debugging and code suggestions but the code is fully written and inferred by us at the end.
1. **Teacher Distillation** – Pretrain the teacher network on ImageNet features.
2. **Algorithm 1 Training** – Distill knowledge into the student and autoencoder.
3. **PUAD Inference** – Combine picturable and unpicturable anomaly detection via Mahalanobis scoring.

## How It Works

### Stage 1 — Teacher Distillation
A small CNN teacher (`PDN_S`, 384 channels) is trained to mimic a frozen **WideResNet101_2** feature extractor. The teacher learns rich patch-level descriptors cheaply. Channel-wise normalization statistics (μ, σ) are computed and saved alongside the teacher weights.

### Stage 2 — Student + Autoencoder Training (Algorithm 1)
Given only **normal images**, a student network and autoencoder are jointly trained:

| Component | Role |
|-----------|------|
| **Student** (`PDN_S`, 768 channels) | Mimics teacher on normal images; penalized for activating on ImageNet images |
| **Autoencoder** | Reconstructs teacher features from augmented normal images |
| **Student-AE channels** | Student's second-half channels are trained to match AE output |

At inference, anomalies cause high reconstruction error in either map → high anomaly score.

Validation quantiles (q₀.₉₀, q₀.₉₉₅) are computed to normalize anomaly maps consistently across categories.

### Stage 3 — PUAD Inference
PUAD combines two complementary detection signals:

| Signal | Type | Method |
|--------|------|--------|
| **Picturable** | Structural defects (scratches, dents) | Pixel-wise student/AE error maps |
| **Unpicturable** | Logical anomalies (wrong assembly) | Mahalanobis distance on global feature vectors |

Final score = normalized EfficientAD score + normalized Mahalanobis distance.

---

## Project Structure

```
EfficientAD/
├── efficientad_teacher_training_only.ipynb   # Stage 1: Teacher training
├── EfficientAD_Algorithm1_Training.ipynb     # Stage 2: Student + AE training
├── PUAD_Implementation_2.ipynb               # Stage 3: PUAD inference + evaluation
├── Save_checkpoints/
│   └── teacher_final.pth                     # Saved teacher + μ/σ
├── model_<category>/
│   ├── student.pt
│   ├── autoencoder.pt
│   ├── mu.pt
│   ├── sigma.pt
│   └── quantile.pt
└── mvtec_loco_anomaly_detection/
    └── <category>/
        ├── train/good/
        ├── validation/good/
        └── test/
            ├── good/
            ├── logical_anomalies/
            └── structural_anomalies/
```

---

## Notebooks
- [efficientad_teacher_training_only.ipynb](efficientad_teacher_training_only.ipynb) – trains the teacher using WideResNet101_2 features, supports checkpoint resume, and saves `teacher_final.pth` with mu,sigma $ statistics.
- [EfficientAD_Algorithm1_Training.ipynb](EfficientAD_Algorithm1_Training.ipynb) – fine-tunes student + autoencoder, logs losses, computes quantiles (q_{0.90},q_{0.995}), and writes `student.pt`, `autoencoder.pt`, `mu.pt`, `sigma.pt`, `quantile.pt`.
- [PUAD_Implementation_2.ipynb](PUAD_Implementation_2.ipynb) – loads trained weights, fits PUAD statistics on normal data, evaluates AUROC, renders anomaly heatmaps, and profiles latency/FLOPs.

## Prerequisites
- PyTorch with CUDA, torchvision, tqdm, scikit-learn, matplotlib.
- Datasets:
  - ImageNet subset at `./archive/train_combined`. (have combined all the training subfolders) [Imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100)
  - MVTec LOCO category folders under `./mvtec_loco_anomaly_detection/<category>/`. [MVTec Loco](https://www.mvtec.com/company/research/datasets/mvtec-loco)
- Checkpoint directories: `./Save_checkpoints`23211 and per-category `./model_<category>/`.

## Workflow
1. **Train Teacher**  
   Open [efficientad_teacher_training_only.ipynb](efficientad_teacher_training_only.ipynb) → configure dataset paths → run cells 1–10 → obtain `teacher_final.pth`.
2. **Train Student & Autoencoder**  
   Open [EfficientAD_Algorithm1_Training.ipynb](EfficientAD_Algorithm1_Training.ipynb) → update `config` → execute sections 1–12 → collect trained weights and quantiles in the chosen `model_dir`.
3. **Run PUAD Inference**  
   Open [PUAD_Implementation_2.ipynb](PUAD_Implementation_2.ipynb) → set `dataset_name` and paths → run training/validation/testing cells → inspect AUROC summaries, heatmaps, and efficiency metrics.

## Outputs
- `./Save_checkpoints/teacher_final.pth`  
- `./model_<category>/student.pt`, `autoencoder.pt`, `mu.pt`, `sigma.pt`, `quantile.pt`  
- Diagnostic plots (loss curves, teacher validation histograms) and heatmaps stored alongside notebooks.

## Notes
- Enable `torch.backends.cudnn.benchmark = True` on CUDA devices for faster inference.
- Quantile normalization scales anomaly maps
- Efficiency profiling reports latency, throughput, parameter counts, FLOPs, and GPU memory for both EfficientAD and PUAD stages.
