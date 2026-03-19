# 🚀 Deployment Guide

## Skin Lesion Classification — Deployment Options

---

## Option 1: Google Colab (Development)

1. Open the notebook in Colab
2. Run all cells from Cell 1 to Cell 20
3. Gradio app launches with a temporary public URL (72 hours)

**Best for:** Testing, development, retraining

---

## Option 2: Local Machine

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run
```bash
cd skin-lesion-classification
python app/gradio_app.py
```

The app downloads the model from HuggingFace automatically on first run.

**Best for:** Local testing, demos

---

## Option 3: HuggingFace Spaces (Production — Free)

The app is deployed at:
**https://huggingface.co/spaces/code-with-zeeshan/skin-lesion-classifier**

### Files on HuggingFace:
- `app.py` — Gradio application
- `model_b3.keras` — Trained model
- `final_config.json` — Configuration
- `requirements.txt` — Dependencies
- `README.md` — Space description

**Best for:** Permanent public hosting, sharing

---

## Model Files

| File | Size | Location |
|---|---|---|
| model_b3.keras | ~96.4 MB | HuggingFace Space |
| model_b0.keras | ~33.4 MB | HuggingFace Space |
| final_config.json | 1 KB | GitHub + HuggingFace |

---

## Configuration

`final_config.json` contains:
```json
{
  "model_name": "Skin Lesion Classifier V2",
  "best_method": "B3 + TTA",
  "models": {
    "model_b0": "model_b0.keras",
    "model_b3": "model_b3.keras"
  },
  "img_size": 300,
  "threshold": 0.37999999999999995,
  "tta_rounds": 10,
  "classes": {
    "0": "benign",
    "1": "malignant"
  },
  "final_metrics": {
    "auc": 0.9353,
    "accuracy": 0.8379,
    "recall_malignant": 0.9467,
    "precision_malignant": 0.7573,
    "f1_malignant": 0.8415,
    "missed_cancers": 16,
    "false_alarms": 91
  }
}
```

**Important:** Image preprocessing must NOT include `rescale=1./255`.
EfficientNet has built-in preprocessing expecting [0, 255] pixel values.


### Verify docs/ Content

```python
# Check docs folder
print("📁 docs/ folder contents:")
for f in os.listdir(f'{GITHUB_DIR}/docs'):
    size = os.path.getsize(os.path.join(f'{GITHUB_DIR}/docs', f)) / 1024
    print(f"  📄 {f} ({size:.1f} KB)")
```
