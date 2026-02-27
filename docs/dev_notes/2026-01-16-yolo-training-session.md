# YOLOv11 Training Session - January 15-16, 2026

## Overview
Training YOLOv11s for document layout detection using DocLayNet + custom dataset.

## Training History

### V23 (Epoch 37) - Best Model
- **mAP50**: 40.8%
- **mAP50-95**: 33.6%
- **Training**: True resume from V14 checkpoint

### V29 (Epoch 50 attempt) - Failed Improvement
- **mAP50**: 36.2% (decreased)
- **Issue**: Weights-only resume with default lr=0.01 destabilized the model
- **Lesson**: Must use lower LR when fine-tuning completed checkpoints

## Key Technical Findings

### YOLO Resume Modes

| Mode | Use Case | Settings |
|------|----------|----------|
| `resume=True` | Interrupted training | Restores optimizer, LR, epoch |
| `resume=False` | Completed training | Loads weights only |

### True Resume Failures

YOLO's true resume (`resume=True`) failed because:
1. Training completed at epoch 37/37
2. Checkpoint marked as "finished" internally
3. Updating `train_args.epochs` in checkpoint didn't help
4. YOLO's resume logic still detected training as complete

### Correct Fine-Tuning Approach

For extending COMPLETED training:
```python
model.train(
    lr0=0.0001,          # 100x lower than default
    warmup_epochs=0.5,   # Minimal warmup
    resume=False,
)
```

## Per-Class Performance (V23)

| Class | mAP50 | Notes |
|-------|-------|-------|
| Text | 82.9% | Excellent |
| Table | 63.2% | Good |
| Title | 53.8% | Moderate |
| List | 44.6% | Moderate |
| Footer | 29.6% | Low |
| Figure | 24.0% | Low |
| Header | 15.2% | Poor - confused with Text/Title |
| Caption | 13.6% | Poor - confused with Footer |

## Scripts Created

1. `kaggle_resume_v15.py` - True resume attempt (failed)
2. `kaggle_resume_v16.py` - Weights-only resume (destabilized)
3. `kaggle_resume_v17.py` - Fine-tuning approach (pending)

## Next Steps

1. Run V17 with fine-tuning settings (lr=0.0001, warmup=0.5)
2. Expected: mAP50 > 45%
3. If still low, consider more training data for weak classes
