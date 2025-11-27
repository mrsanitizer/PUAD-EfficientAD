# EfficientAD Training & PUAD Inference

## Overview
This workspace contains a three-stage pipeline for EfficientAD anomaly detection:
LLMs were used for debugging and code suggestions but the code is fully written and inferred by us at the end.
1. **Teacher Distillation** – Pretrain the teacher network on ImageNet features.
2. **Algorithm 1 Training** – Distill knowledge into the student and autoencoder.
3. **PUAD Inference** – Combine picturable and unpicturable anomaly detection via Mahalanobis scoring.

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
