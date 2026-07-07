import gradio as gr
import numpy as np
import joblib
import librosa
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Load model and feature names
model = joblib.load('../models/best_lgbm.pkl')
feature_names = joblib.load('../models/feature_names.pkl')
explainer = shap.TreeExplainer(model)

def load_audio(path, sr=22050, max_duration=5.0):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_duration)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.3:
        return None
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max()
    return y

def extract_features(path, sr=22050, n_mfcc=13):
    y = load_audio(path, sr=sr)
    if y is None:
        return None
    feats = []
    def agg(arr):
        return np.concatenate([arr.mean(axis=1), arr.std(axis=1), arr.max(axis=1)])
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    for arr in [mfcc, mfcc_d, mfcc_d2]:
        feats.append(agg(arr))
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    sco = librosa.feature.spectral_contrast(y=y, sr=sr)
    for arr in [sc, sb, sro]:
        feats.append(agg(arr))
    feats.append(agg(sco))
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    for arr in [zcr, rms]:
        feats.append(agg(arr))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.append(agg(chroma))
    feats.append(np.array([len(y) / sr]))
    return np.concatenate(feats).astype(np.float32)

def predict(audio_path):
    if audio_path is None:
        return "No audio provided.", None

    vec = extract_features(audio_path)
    if vec is None:
        return "Audio too short or unreadable. Please upload a longer clip.", None

    prob = model.predict_proba([vec])[0]
    covid_prob   = prob[1]
    healthy_prob = prob[0]

    if covid_prob >= 0.5:
        label = "⚠️ COVID-19 Likely"
    else:
        label = "✅ Healthy Likely"

    result = f"""### {label}

| Class | Probability |
|-------|------------|
| Healthy | {healthy_prob:.1%} |
| COVID-19 | {covid_prob:.1%} |

---
*AUROC: 0.6179 on holdout. Research demo — not a medical diagnostic tool.*"""

    # SHAP bar plot — saved as image
    shap_vals = explainer.shap_values(vec.reshape(1, -1))
    indices   = np.argsort(np.abs(shap_vals[0]))[-15:]
    top_shap  = shap_vals[0][indices]
    top_names = [feature_names[i] for i in indices]
    colors    = ['tomato' if v > 0 else 'steelblue' for v in top_shap]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names, top_shap, color=colors, alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP value (positive = toward COVID-19)')
    ax.set_title('Top 15 features driving this prediction')
    plt.tight_layout()

    # Save to temp file and return path
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close()

    return result, tmp.name

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload or record a cough"),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Image(label="Feature attribution (SHAP)", type="filepath")
    ],
    title="🫁 Cough-based COVID-19 Detector",
    description="""Upload a cough recording (wav, mp3, ogg) and the model will predict COVID-19 likelihood
using 190 hand-crafted audio features. Built with LightGBM + SHAP on the COUGHVID dataset.

**⚠️ Research demo only — not a medical diagnostic tool.**""",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)