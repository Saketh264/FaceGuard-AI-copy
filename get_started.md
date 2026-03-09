# FaceGuard Training Get Started Guide

## 1. Purpose of this guide
This document translates your project planning documents in [docs/](docs/) (proposal, RTM, architecture, data-flow) into a practical model-training execution plan for this repository.

Your planning direction is:
- MVP 1: ResNet-based detector.
- MVP 2: Hybrid CNN-ViT detector.
- Privacy-preserving operation (no image retention, EXIF removal, secure transport).
- Fairness-aware evaluation across demographic groups.

This repository is the AI-Face fairness benchmark codebase. It already provides multiple trainable deepfake detectors and fairness evaluation logic, but it does not fully implement your exact planned architecture path out of the box.

## 2. What is already provided in this repository

### 2.1 Core training entry points
- [training/train_test.py](training/train_test.py)
  - Main training script for detector classes registered under [training/detectors/](training/detectors/).
  - Supports: `xception`, `efficientnet`, `core`, `ucf`, `srm`, `f3net`, `spsl`, `daw_fdd`, `dag_fdd`, `fair_df_detector`.
- [training/train_test_vit.py](training/train_test_vit.py)
  - ViT-B/16 training pipeline.
- [training/train_test_clip.py](training/train_test_clip.py)
  - UnivFD-style CLIP-based training pipeline.

### 2.2 Dataset loaders
- [training/dataset/datasets_train.py](training/dataset/datasets_train.py)
- [training/dataset/pair_dataset.py](training/dataset/pair_dataset.py)

### 2.3 Fairness metrics
- [training/fairness_metrics.py](training/fairness_metrics.py)
- [training/fairness_metrics_srm.py](training/fairness_metrics_srm.py)

### 2.4 Important reality check
Out of the box, this repo does **not** provide:
- A ready-to-run `resnet` detector option in [training/train_test.py](training/train_test.py).
- A ready-to-run hybrid CNN-ViT detector implementation.

So for your exact project roadmap (`ResNet -> CNN-ViT`), you will need custom code (details below).

## 3. What is not locally included and must be downloaded

### 3.1 Dataset files
The repo [dataset/](dataset/) folder is a placeholder. You must download the dataset using the links in [README.md](README.md) and place CSV metadata and image files accordingly.

### 3.2 Pretrained weights
Most Xception-based detectors require:
- [training/pretrained/xception-b5690688.pth](training/pretrained/xception-b5690688.pth)

You must download and place this file manually before training Xception-family methods.

## 4. Team training strategy aligned with your plan

Use a 3-track strategy:

### Track A (Immediate baseline, no custom model code)
Goal: get working reproducible results quickly.
- Train `xception` (fastest path in current repo).
- Train `vit` via `train_test_vit.py`.
- Optional: `efficientnet` for additional CNN baseline.

### Track B (Planned MVP 1: ResNet)
Goal: align with proposal.
- Add your own `resnet_detector.py` under `training/detectors/`.
- Register it so `train_test.py --model resnet` works.

### Track C (Planned MVP 2: Hybrid CNN-ViT)
Goal: implement your advanced target architecture.
- Build a custom hybrid model module.
- Add a dedicated training script or detector class.

## 5. Step-by-step setup and first training run

## Step 1: Environment setup
From repository root:

```bash
conda create -n FairnessBench python=3.9.0
conda activate FairnessBench
pip install -r [requirements.txt](requirements.txt)
```

Notes:
- Scripts currently assume CUDA and set device as `cuda:0` directly.
- The CLI argument `--gpu` exists but is not used in the scripts.

## Step 2: Dataset download and placement
1. Download AI-Face image archives and annotation files from [README.md](README.md) links.
2. Extract image archives.
3. Place `train.csv` and `test.csv` under [dataset/](dataset/).
4. Ensure each CSV row contains valid image paths accessible from your training machine.

Minimum expected columns used by training/evaluation code:
- `Image Path`
- `Target`
- `Intersection`

Additional expected columns in some flows:
- `Predicted Age` (used by test subgroup evaluation in `ImageDataset_Test`).
- `Specific` (used for pair training with UCF/Fair-DF style methods).

## Step 3: Pretrained checkpoints
For Xception-family methods:
1. Download Xception pretrained weight file from README link.
2. Place it at:
   - [training/pretrained/xception-b5690688.pth](training/pretrained/xception-b5690688.pth)

## Step 4: First baseline training (recommended)
From repository root:

```bash
cd training
python train_test.py --model xception --datapath ../dataset/ --test_datapath ../dataset/test.csv --savepath ../results
```

What happens:
- Trains for 10 epochs (hardcoded).
- Saves checkpoints to `training/checkpoints/<model>/`.
- Runs fairness evaluation each epoch.
- Writes logs to `training/checkpoints/<model>/log_training.txt`.

## Step 5: ViT baseline training

```bash
cd training
python train_test_vit.py --datapath ../dataset/ --test_datapath ../dataset/test.csv --savepath ../results
```

## Step 6: Optional additional baselines

### EfficientNet baseline
```bash
cd training
python train_test.py --model efficientnet --datapath ../dataset/ --test_datapath ../dataset/test.csv --savepath ../results
```

### Pair-based methods (UCF / fair_df_detector)
Use only if you have pair CSV files prepared:
- [dataset/train_fake_spe.csv](dataset/train_fake_spe.csv)
- [dataset/train_real.csv](dataset/train_real.csv)

Then:

```bash
cd training
python train_test.py --model ucf --dataset_type pair --datapath ../dataset/ --test_datapath ../dataset/test.csv --savepath ../results
```

## 6. Should your team create your own training script?

Short answer:
- For immediate benchmark training: **No**, you can use provided scripts.
- For your planned architecture path (ResNet and hybrid CNN-ViT): **Yes**, custom implementation is required.

Reason:
- Existing scripts do not expose `resnet` detector option.
- No hybrid CNN-ViT detector/trainer is implemented in this repository.

## 7. How to implement MVP 1 (ResNet) in this repo

Recommended approach: extend `train_test.py` detector registry.

## Step 1: Create detector file
Create [training/detectors/resnet_detector.py](training/detectors/resnet_detector.py) with this shape:
- Class extends `AbstractDetector`.
- Register with `@DETECTOR.register_module(module_name='resnet')`.
- `build_backbone()` uses `BACKBONE['resnet50']` (or `resnet34`).
- `build_loss()` uses BCE loss (`LOSSFUNC['bce']`).
- `forward()` returns `{'cls': pred}`.
- `get_train_metrics()` uses `calculate_metrics_for_train`.

## Step 2: Register it
Update [training/detectors/__init__.py](training/detectors/__init__.py):
- Add `from .resnet_detector import ResNetDetector`.

## Step 3: Train it

```bash
cd training
python train_test.py --model resnet --datapath ../dataset/ --test_datapath ../dataset/test.csv --savepath ../results
```

## Step 4: Report against RTM
Capture:
- Accuracy, AUC, EER, AP.
- Fairness gaps from fairness metrics printout.
- Subgroup performance stability.

## 8. How to implement MVP 2 (Hybrid CNN-ViT)

No native script exists, so implement one of these patterns.

### Option A: Dedicated script (recommended for clarity)
Create [training/train_test_cnn_vit.py](training/train_test_cnn_vit.py):
- Reuse data loaders from existing scripts.
- Add model from [training/models/cnn_vit_hybrid.py](training/models/cnn_vit_hybrid.py).
- Keep same logging/checkpoint/fairness evaluation flow as [training/train_test_vit.py](training/train_test_vit.py).

### Option B: New detector class integrated into [training/train_test.py](training/train_test.py)
Create [training/detectors/cnn_vit_detector.py](training/detectors/cnn_vit_detector.py):
- Keeps one CLI entry point ([training/train_test.py](training/train_test.py)).
- Better long-term integration with existing detector registry.

### Suggested hybrid architecture (practical and incremental)
1. CNN feature extractor (ResNet34 or ResNet50 backbone).
2. Patch/token projection from CNN feature map.
3. Transformer encoder blocks on tokens.
4. CLS or pooled token head for binary classification.
5. BCEWithLogits loss.

### Training plan for hybrid model
1. Warm-start with CNN pretrained weights.
2. Stage 1: freeze partial CNN, train transformer head.
3. Stage 2: unfreeze all, low LR fine-tuning.
4. Evaluate fairness and subgroup gaps after each epoch.

## 9. Data and CSV validation checklist (mandatory)
Before long training runs, validate CSVs:

- No missing image paths.
- All paths exist on disk.
- `Target` in `{0,1}`.
- `Intersection` values consistent with your annotation scheme.
- `Predicted Age` exists if you evaluate age subgroup metrics.
- For pair mode: `Specific` column exists in fake CSV.

## 10. Known code constraints and caveats

1. Device selection
- Scripts hardcode `cuda:0`.
- If running CPU or different GPU, adjust code accordingly.

2. Epoch count
- Training scripts currently hardcode `num_epochs=10`.
- Make it a CLI argument if your experiments need longer runs.

3. Resume training support
- [training/train_test.py](training/train_test.py) has partial continue support.
- [training/train_test_vit.py](training/train_test_vit.py) and [training/train_test_clip.py](training/train_test_clip.py) define continue args but do not load checkpoints.

4. Pair-mode documentation gap
- README focuses on [dataset/train.csv](dataset/train.csv)/[dataset/test.csv](dataset/test.csv) but pair models require extra CSV files.

5. Privacy scope
- This repository is mainly model training/evaluation.
- Privacy-by-design items from your proposal (ephemeral uploads, EXIF stripping, no retention, HTTPS, auth) belong primarily in backend/inference service implementation.

## 11. Recommended team execution plan (aligned to your proposal phases)

### Phase 1: Baseline validation (1-2 weeks)
- Get dataset and CSV schema stable.
- Train Xception and ViT baselines using existing scripts.
- Record metrics and fairness outputs.

### Phase 2: ResNet MVP (1-2 weeks)
- Implement [training/detectors/resnet_detector.py](training/detectors/resnet_detector.py).
- Train and compare against Xception/ViT baselines.
- Add Grad-CAM pipeline for explainability-ready outputs.

### Phase 3: Hybrid CNN-ViT MVP (2-4 weeks)
- Implement hybrid model and trainer.
- Run ablations (CNN only vs ViT only vs Hybrid).
- Benchmark fairness and robustness.

### Phase 4: Production integration preparation
- Export best checkpoint.
- Integrate inference API (FastAPI).
- Implement privacy-preserving request lifecycle:
  - EXIF stripping,
  - in-memory processing,
  - automatic deletion,
  - HTTPS,
  - logging without raw-image retention.

## 12. Minimum experiment table template (for weekly reporting)
Use this schema in your project tracker:

- Experiment ID
- Model (`xception`, `vit`, `resnet`, `cnn_vit_hybrid`)
- Dataset split version
- Epochs / LR / Batch size
- Overall Accuracy / AUC / EER / AP
- Fairness gaps (gender, skin tone, age)
- Notes (failure modes, subgroup weakness, next action)

## 13. Definition of Done for training milestone
A model training milestone is complete when all are true:

1. Reproducible run command and config are documented.
2. Checkpoint and logs are saved and versioned.
3. Overall metrics and subgroup fairness metrics are reported.
4. Explainability output path (Grad-CAM or equivalent) is validated.
5. Chosen model is ready for backend inference integration.

## 14. Final recommendation for your current situation

Start immediately with existing scripts to establish baseline evidence for your supervisor:
1. `xception` baseline via [training/train_test.py](training/train_test.py).
2. `vit` baseline via [training/train_test_vit.py](training/train_test_vit.py).

Then implement your planned architecture roadmap:
3. Add [training/detectors/resnet_detector.py](training/detectors/resnet_detector.py) for MVP 1.
4. Build [training/train_test_cnn_vit.py](training/train_test_cnn_vit.py) (or detector) for MVP 2.

This approach gives you both:
- fast progress using provided code, and
- strong alignment with your proposal commitments.
