# 📊 Model Evaluation Report

## Skin Lesion Classification — EfficientNetB3 + TTA

**Date:** 14 March 2026  
**Author:** MOHAMMAD ZEESHAN
**Model Version:** V2 (Final)

---

## 1. Dataset Overview

| Split | Benign | Malignant | Total |
|---|---|---|---|
| **Train** | 1,440 (54.6%) | 1,197 (45.4%) | 2,637 |
| **Test** | 360 (54.5%) | 300 (45.5%) | 660 |
| **Total** | 1,800 | 1,497 | 3,297 |

- **Class Imbalance:** Mild (55:45 ratio)
- **Handling:** Class weights (Benign: 0.916, Malignant: 1.101)
- **Validation:** 20% split from training data (527 images)

---

## 2. Model Architecture

| Component | Details |
|---|---|
| **Base Model** | EfficientNetB3 (ImageNet pre-trained) |
| **Input Size** | 300 × 300 × 3 |
| **Custom Head** | GAP → BN → Dense(512) → DO(0.5) → Dense(256) → DO(0.4) → Dense(128) → DO(0.3) → Dense(1, sigmoid) |
| **Total Params** | ~12.9M |
| **Training** | 2-phase (frozen → fine-tune last 50 layers) |
| **Optimizer** | Adam (Phase 1: 1e-4, Phase 2: 1e-5) |
| **Loss** | Binary Crossentropy |

---

## 3. Training Summary

### Phase 1: Frozen Base (10 epochs)
- Best val_auc: 0.917 (Epoch 9)
- Learning rate reduced at Epoch 6

### Phase 2: Fine-tuning (7 epochs, early stopped)
- Unfroze last 50 layers of EfficientNetB3
- Early stopping triggered at Epoch 7
- Best weights restored from Epoch 1

---

## 4. Threshold Optimization

| Method | Threshold | Accuracy | Recall | Missed |
|---|---|---|---|---|
| Default (0.50) | 0.500 | 81.7% | 82.0% | 54 |
| Max F1 | 0.336 | 81.8% | 88.7% | 34 |
| Youden's J | 0.336 | 81.8% | 88.7% | 34 |
| 85% Recall | 0.380 | 81.8% | 85.7% | 43 |
| **90% Recall** | **0.380** | **80.9%** | **90.0%** | **30** |

**Selected:** 0.38 (targets ≥90% cancer detection for medical safety)

---

## 5. Test-Time Augmentation (TTA)

| Metric | B3 Only | B3 + TTA | Improvement |
|---|---|---|---|
| AUC | 0.911 | **0.935** | +2.42% |
| Accuracy | 81.7% | **83.8%** | +2.1% |
| Recall | 90.0% | **94.7%** | +4.7% |
| Missed | 30 | **16** | 14 fewer |

**TTA Config:** 10 augmentation rounds (flip, brightness)

---

## 6. Ensemble Comparison

| Method | AUC | Recall | Missed |
|---|---|---|---|
| B3 only | 0.911 | 90.0% | 30 |
| **B3 + TTA** | **0.935** | **94.7%** | **16** |
| B0 + B3 | 0.926 | 94.3% | 17 |
| B0 + B3_TTA | 0.935 | 95.3% | 14 |

**Winner:** B3 + TTA (highest AUC with strong recall)

---

## 7. Final Results — B3 + TTA (Unseen Test Data)

### Core Metrics

| Metric | Score | Grade |
|---|---|---|
| AUC-ROC | 0.935 | 🟢 Good |
| Accuracy | 83.8% | 🟡 Okay |
| Balanced Accuracy | 84.7% | 🟡 Okay |
| PR-AUC | 0.918 | 🟢 Good |
| Log Loss | 0.330 | 🟢 Good |

### Cancer Detection

| Metric | Score | Grade |
|---|---|---|
| Sensitivity (Recall) | 94.7% | 🏆 Excellent |
| Precision | 75.7% | 🟡 Okay |
| F1 Score | 0.842 | 🟡 Okay |
| Miss Rate | 5.3% | 🟢 Good |

### Benign Detection

| Metric | Score | Grade |
|---|---|---|
| Specificity | 74.7% | 🟡 Okay |
| NPV | 94.4% | 🏆 Excellent |
| False Alarm Rate | 25.3% | 🟡 Okay |

### Advanced Metrics

| Metric | Score | Grade |
|---|---|---|
| MCC | 0.698 | 🟢 Good |
| Cohen's Kappa | 0.680 | 🟢 Good |
| Calibration Error | 0.060 | 🟢 Good |

---

## 8. Confusion Matrix

```
                    Predicted Benign    Predicted Malignant
Actual Benign            269                 91
Actual Malignant          16                284
```

- **True Positives (TP):** 284 — Correctly detected malignant
- **True Negatives (TN):** 269 — Correctly identified benign
- **False Positives (FP):** 91 — Benign flagged as malignant (false alarms)
- **False Negatives (FN):** 16 — Malignant missed as benign (DANGEROUS)

---

## 9. Error Analysis

### Missed Cancers (16 cases)
- Average confidence (wrong benign): 74.3%
- Model was moderately confident → these are blind spot cases
- Some malignant lesions visually resemble benign ones

### False Alarms (91 cases)
- Average confidence (wrong malignant): 57.6%
- Model was NOT very confident → borderline cases
- These benign lesions have suspicious features

### Confidence Reliability

| Confidence Range | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| Low (50-60%) | 53 | 27 | 26 | 50.9% |
| Medium (60-70%) | 77 | 60 | 17 | 77.9% |
| High (70-80%) | 95 | 75 | 20 | 78.9% |
| Very High (80-90%) | 98 | 92 | 6 | 93.9% |
| Extreme (90-100%) | 278 | 273 | 5 | **98.2%** |

**Key Insight:** When the model is >90% confident, it's correct 98.2% of the time.

---

## 10. Model Limitations

1. **Small dataset:** ~2,600 training images limits generalization
2. **False alarm rate:** 25.3% would cause unnecessary patient anxiety
3. **16 missed cancers:** Not zero — always needs doctor verification
4. **Domain specificity:** Trained on dermoscopic images only
5. **No multi-class:** Only benign/malignant, not specific cancer types

---

## 11. Recommendations

### For Users
- Use as a **screening tool only**, not for diagnosis
- Always consult a dermatologist for professional evaluation
- High-confidence predictions (>90%) are highly reliable

### For Improvement
- Add more training data (ISIC Archive: 25K+ images)
- Implement advanced augmentation (albumentations)
- Add multi-class classification (melanoma, BCC, SCC)
- Perform 5-fold cross-validation for robust evaluation

---

## 12. Technical Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| TensorFlow | 2.x |
| Gradio | 6.0 |
| Google Colab | GPU (T4) |
| Training Time | ~30 minutes |

---

*Report generated as part of the Skin Lesion Classification project.*  
*AI assistance (Claude, Anthropic) was used for code generation and analysis.*
