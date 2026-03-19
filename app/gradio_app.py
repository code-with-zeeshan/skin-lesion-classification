"""
🏥 Skin Lesion Classification — Gradio App
Models downloaded from HuggingFace on first run.
Compatible with Gradio 6.0+
"""

import os
import json
import numpy as np
import cv2
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

# ════════════════════════════════════════════
# 📥 DOWNLOAD MODEL FROM HUGGINGFACE
# ════════════════════════════════════════════

HF_REPO_ID = "code-with-zeeshan/skin-lesion-classifier"

def download_from_hf():
    """Download model and config from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model_b3.keras",
                                  repo_type="space")
    config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="final_config.json",
                                   repo_type="space")
    return model_path, config_path

print("📥 Downloading model from HuggingFace...")
MODEL_PATH, CONFIG_PATH = download_from_hf()

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

IMG_SIZE = config["img_size"]
THRESHOLD = config["threshold"]

model = load_model(MODEL_PATH)
print(f"✅ Model loaded | Threshold: {THRESHOLD} | IMG_SIZE: {IMG_SIZE}")

# ════════════════════════════════════════════
# 🧠 CORE FUNCTIONS
# ════════════════════════════════════════════

def generate_gradcam(mdl, img_arr):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        inputs=mdl.input,
        outputs=[mdl.get_layer('top_activation').output, mdl.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_arr)
        loss = preds[0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def predict_fast(mdl, img_batch):
    """⚡ Fast Mode: Single prediction."""
    return float(mdl.predict(img_batch, verbose=0)[0][0])


def predict_tta(mdl, img_batch, n_augments=10):
    """🎯 Best Mode: TTA prediction."""
    preds = [mdl.predict(img_batch, verbose=0)[0][0]]
    for _ in range(n_augments):
        aug = img_batch.copy()
        if np.random.random() > 0.5:
            aug = np.flip(aug, axis=2)
        if np.random.random() > 0.5:
            aug = np.flip(aug, axis=1)
        aug = np.clip(aug * np.random.uniform(0.85, 1.15), 0, 255)
        preds.append(float(mdl.predict(aug.copy(), verbose=0)[0][0]))
    return np.mean(preds)


# ════════════════════════════════════════════
# 📋 PRECAUTIONS
# ════════════════════════════════════════════

PRECAUTIONS = {
    "benign": {
        "title": "✅ BENIGN — Low Risk",
        "summary": "Lesion appears benign. Regular monitoring recommended.",
        "items": [
            "📋 Monthly self-skin examinations",
            "🔍 Monitor for changes in size, shape, color",
            "📸 Photograph periodically to track changes",
            "🧴 SPF 30+ sunscreen daily",
            "🩺 Dermatologist check-up every 6-12 months",
            "⚠️ See doctor immediately if lesion changes"
        ],
        "extra_title": "⚠️ WHEN TO WORRY:",
        "extra": [
            "🔴 Rapid growth",
            "🔴 Color changes",
            "🔴 Irregular borders",
            "🔴 Bleeding or itching",
            "🔴 New lesions nearby"
        ]
    },
    "malignant": {
        "title": "⚠️ MALIGNANT — High Risk",
        "summary": "IMMEDIATE medical consultation strongly recommended.",
        "items": [
            "🚨 Consult dermatologist IMMEDIATELY",
            "🏥 Schedule biopsy for definitive diagnosis",
            "📋 Do NOT self-treat",
            "📸 Document with photos + size reference",
            "🧬 Ask about genetic testing if family history",
            "☀️ Avoid sun exposure on affected area",
            "🩺 Request full-body skin examination"
        ],
        "extra_title": "🔍 ABCDE RULE — Signs of Melanoma:",
        "extra": [
            "A — Asymmetry",
            "B — Irregular Border",
            "C — Multiple Colors",
            "D — Diameter > 6mm",
            "E — Evolving shape/size"
        ]
    }
}

# ════════════════════════════════════════════
# 🎯 MAIN PREDICTION
# ════════════════════════════════════════════

def predict_skin_lesion(input_image, patient_name, patient_age, patient_gender, mode):
    """Main prediction function with Fast/Best mode toggle."""
    if input_image is None:
        return None, None, "⚠️ Please upload an image."

    patient_name = patient_name if patient_name and patient_name.strip() else "Anonymous"
    patient_age = patient_age if patient_age and patient_age > 0 else "N/A"
    patient_gender = patient_gender or "Not specified"

    # Preprocess
    img_resized = input_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict based on mode
    if mode == "🎯 Best Mode (TTA — Higher Accuracy, Slower)":
        prediction = predict_tta(model, img_batch, n_augments=10)
        mode_label = "🎯 Best Mode (TTA x10)"
        mode_stats = "AUC: 0.935 | Recall: 94.7% | Accuracy: 83.8%"
    else:
        prediction = predict_fast(model, img_batch)
        mode_label = "⚡ Fast Mode (Single)"
        mode_stats = "AUC: 0.911 | Recall: 90.0% | Accuracy: 81.7%"

    is_malignant = prediction >= THRESHOLD
    label = "MALIGNANT" if is_malignant else "BENIGN"
    confidence = float(prediction if is_malignant else 1 - prediction)

    # Grad-CAM
    heatmap = generate_gradcam(model, img_batch)
    hm = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(np.array(img_resized), (IMG_SIZE, IMG_SIZE))
    overlay = np.uint8(hm_color * 0.4 + img_np * 0.6)

    # Output image with border
    border_color = (255, 0, 0) if is_malignant else (0, 180, 0)
    out = cv2.copyMakeBorder(img_np, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=border_color)
    bar = np.zeros((50, out.shape[1], 3), dtype=np.uint8)
    bar[:] = border_color
    cv2.putText(bar, f"{label} ({confidence:.1%})", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    output_img = np.vstack([bar, out])

    # Grad-CAM image with border
    gc = cv2.copyMakeBorder(overlay, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 165, 0))
    bar_gc = np.zeros((50, gc.shape[1], 3), dtype=np.uint8)
    bar_gc[:] = (255, 165, 0)
    cv2.putText(bar_gc, "GRAD-CAM: Model Focus", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    gradcam_img = np.vstack([bar_gc, gc])

    # Build report
    key = "malignant" if is_malignant else "benign"
    p = PRECAUTIONS[key]
    report = [
        "=" * 50,
        "🏥 SKIN LESION ANALYSIS REPORT",
        "=" * 50, "",
        f"👤 {patient_name} | Age: {patient_age} | {patient_gender}",
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"⚙️ {mode_label}",
        f"   Performance: {mode_stats}", "",
        "─" * 50,
        f"🔬 Diagnosis: {label}",
        f"   Confidence: {confidence:.2%}",
        f"   Risk Level: {'HIGH RISK ⚠️' if is_malignant else 'LOW RISK ✅'}",
        f"   Score: {prediction:.4f} | Threshold: {THRESHOLD:.4f}", "",
        "─" * 50,
        f"{'🔴' if is_malignant else '🟢'} {p['title']}",
        f"📝 {p['summary']}", "",
        "🛡️ PRECAUTIONS:",
        *[f"  {i+1}. {item}" for i, item in enumerate(p['items'])], "",
        f"{p['extra_title']}",
        *[f"  {item}" for item in p['extra']], "",
        "─" * 50,
        "🔥 GRAD-CAM: 🔴 Red=Focus | 🟡 Yellow=Moderate | 🔵 Blue=Ignore", "",
        "═" * 50,
        "⚕️ DISCLAIMER: Educational/screening only.",
        "   Always consult a qualified dermatologist.",
        "═" * 50
    ]

    return output_img, gradcam_img, "\n".join(report)


# ════════════════════════════════════════════
# 🚀 GRADIO INTERFACE (Gradio 6.0 Compatible)
# ════════════════════════════════════════════

# Detect Gradio version for compatibility
GRADIO_VERSION = int(gr.__version__.split('.')[0])
print(f"📦 Gradio version: {gr.__version__}")

def build_interface():
    """Build Gradio interface compatible with v5 and v6."""

    with gr.Blocks(title="🏥 Skin Lesion Classifier") as demo:

        # Header
        gr.Markdown("""
        # 🏥 Skin Lesion Classification — AI Screening Tool
        ### EfficientNetB3 + Grad-CAM | AUC: 0.935 | Recall: 94.7%

        > ⚠️ **Disclaimer:** Educational/screening purposes only.
        > Always consult a qualified dermatologist.

        ---
        """)

        with gr.Row():

            # ════════ LEFT PANEL ════════
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload & Patient Info")

                input_image = gr.Image(type="pil", label="📷 Skin Lesion Image", height=300)

                gr.Markdown("### 👤 Patient Details")
                patient_name = gr.Textbox(label="Full Name", placeholder="Patient name...",
                                           max_lines=1)
                patient_age = gr.Number(label="Age", minimum=0, maximum=120, precision=0)
                patient_gender = gr.Dropdown(
                    label="Gender",
                    choices=["Male", "Female", "Other", "Prefer not to say"]
                )

                gr.Markdown("### ⚙️ Prediction Mode")
                mode_selector = gr.Radio(
                    label="Select Mode",
                    choices=[
                        "⚡ Fast Mode (Single Prediction — Quick)",
                        "🎯 Best Mode (TTA — Higher Accuracy, Slower)"
                    ],
                    value="⚡ Fast Mode (Single Prediction — Quick)"
                )

                gr.Markdown("""
                | Mode | AUC | Recall | Accuracy | Speed |
                |---|---|---|---|---|
                | ⚡ **Fast** | 0.911 | 90.0% | 81.7% | ~2 sec |
                | 🎯 **Best** | 0.935 | 94.7% | 83.8% | ~20 sec |
                """)

                with gr.Row():
                    predict_btn = gr.Button("🔬 Analyze Lesion", variant="primary",
                                             size="lg")
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary",
                                           size="lg")

            # ════════ RIGHT PANEL ════════
            with gr.Column(scale=1):
                gr.Markdown("## 🔍 Analysis Results")

                with gr.Row():
                    output_image = gr.Image(label="📊 Prediction Result", height=250)
                    gradcam_image = gr.Image(label="🔥 Grad-CAM (Model Focus)", height=250)

                gr.Markdown("### 📋 Detailed Report & Precautions")
                report_output = gr.Textbox(label="📝 Full Analysis Report", lines=22,
                                            max_lines=50)

        # Footer
        gr.Markdown("""
        ---
        ### 🔥 How to Read Grad-CAM
        | Color | Meaning |
        |---|---|
        | 🔴 **Red/Hot** | Model focused here — HIGH importance |
        | 🟡 **Yellow** | Moderate importance |
        | 🔵 **Blue/Cool** | Low importance — model ignored |

        ---
        ### ℹ️ Model Information
        | Detail | Value |
        |---|---|
        | Architecture | EfficientNetB3 + Custom Head |
        | Input Size | 300 × 300 pixels |
        | Technique | Transfer Learning + TTA |
        | Best AUC | 0.935 |
        | Best Recall | 94.7% |

        ---
        > 🏥 **Always consult a medical professional.**
        """)

        # Button actions
        predict_btn.click(
            fn=predict_skin_lesion,
            inputs=[input_image, patient_name, patient_age, patient_gender, mode_selector],
            outputs=[output_image, gradcam_image, report_output]
        )

        clear_btn.click(
            fn=lambda: (None, "", None, None, None, None, "",
                         "⚡ Fast Mode (Single Prediction — Quick)"),
            inputs=[],
            outputs=[input_image, patient_name, patient_age, patient_gender,
                     output_image, gradcam_image, report_output, mode_selector]
        )

    return demo


demo = build_interface()
print("✅ Gradio interface built!")

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
