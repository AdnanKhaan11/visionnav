# VisionNav Training Guide

## Prerequisites
```bash
uv sync --extra train
export WANDB_API_KEY=your_key
export WANDB_PROJECT=visionnav
```

## 3-Stage Training Pipeline

### Stage 1 — GUI Grounding
**Target:** ScreenSpot Acc@0.5 >= 75%
```bash
make train-stage1
python training/evaluation/eval_screenspot.py --model checkpoints/stage1_best --threshold 75.0
```

### Stage 2 — Action Prediction
**Target:** Action type accuracy >= 85%
```bash
make train-stage2
```

### Stage 3 — Multi-Step Planning
**Target:** ScreenSpot >= 79.6%, Mind2Web Step SR >= 44%
```bash
make train-stage3
make eval
```

## Dataset Preparation
```bash
make data-all
```

## Benchmark Performance Targets
| Stage    | ScreenSpot | Mind2Web SR | Reference          |
|----------|-----------|-------------|-------------------|
| Stage 1  | >= 75.0%  | —           | Grounding baseline |
| Stage 2  | >= 79.6%  | >= 44%      | TongUI-3B level    |
| Stage 3  | >= 83.6%  | >= 53%      | TongUI-3B-1M level |
| 7B model | >= 86.0%  | >= 55%      | TongUI-7B-1M level |
