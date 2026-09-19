# 🫁 Cough-based COVID-19 Detector

A classical ML pipeline that detects COVID-19 from cough recordings using hand-crafted audio features, trained on the COUGHVID dataset.


---

## Results

| Metric | CV (5-fold) | Holdout |
|--------|-------------|---------|
| AUROC | 0.6187 ± 0.015 | 0.6179 |
| Macro F1 | — | 0.5380 |

> The value of this project lies in pipeline rigour, not clinical-grade accuracy.

---

## Project structure

```
CoughDetector/
├── data/
│   ├── coughvid/
│   │   ├── audio/              # Raw .webm/.ogg/.wav files
│   │   ├── metadata.csv        # Original COUGHVID metadata
│   │   └── coughvid_clean.csv  # Filtered labels (healthy / COVID-19 only)
│   ├── X_cv.npy                # Feature matrix (16213 × 190)
│   └── y_cv.npy                # Labels
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_features.ipynb       # Feature extraction
│   ├── 03_modelling.ipynb      # Model training, CV, Optuna tuning
│   └── 04_analysis.ipynb       # SHAP, ablation study
├── models/
│   ├── best_lgbm.pkl           # Tuned LightGBM model
│   └── feature_names.pkl       # 190 feature names
├── demo/
│   └── app.py                  # Gradio demo
├── outputs/                    # Plots and results
└── requirements.txt
```

---

## Pipeline

```
Raw audio (.webm/.ogg/.wav)
        │
        ▼
Preprocessing
  • Resample to 22,050 Hz mono
  • Trim silence (top_db=20)
  • Peak normalise
  • Cap at 5 seconds
        │
        ▼
Feature extraction (190 features)
  • 13 MFCCs + Δ + ΔΔ  → mean / std / max  (117 values)
  • Spectral centroid, bandwidth, rolloff   → mean / std / max  (9 values)
  • Spectral contrast (7 bands)             → mean / std / max  (21 values)
  • Zero crossing rate, RMS energy          → mean / std / max  (6 values)
  • Chroma (12 bins)                        → mean / std / max  (36 values)
  • Duration                                                     (1 value)
        │
        ▼
Model comparison (stratified 5-fold CV)
  LogReg · Random Forest · XGBoost · LightGBM · SVM
        │
        ▼
Optuna hyperparameter tuning (50 trials, LightGBM)
        │
        ▼
Final evaluation on 15% holdout (touched once)
        │
        ▼
SHAP feature importance + feature ablation study
        │
        ▼
Gradio demo → Hugging Face Spaces
```

---

## Dataset

**COUGHVID** ([Zenodo](https://zenodo.org/record/7024894))
- 34,434 crowdsourced cough recordings
- Labels: `healthy`, `COVID-19`, `symptomatic`, `unknown`
- After filtering to binary labels: **16,791 usable recordings**
- Class split: 92.2% healthy / 7.8% COVID-19

Labels are self-reported — recording quality varies significantly across devices and environments. This is the primary source of noise in the dataset.

---

## Model

**LightGBM** with Optuna-tuned hyperparameters:

```python
{
    'n_estimators': 188,
    'num_leaves': 38,
    'learning_rate': 0.0238,
    'min_child_samples': 29,
    'subsample': 0.816,
    'colsample_bytree': 0.617,
    'class_weight': 'balanced'
}
```

`class_weight='balanced'` compensates for the 92/8 class imbalance by penalising COVID-19 misclassifications ~11× more heavily.

---

## Feature importance (SHAP)

Top features driving model predictions:

| Rank | Feature | Interpretation |
|------|---------|----------------|
| 1 | `mfcc_12_mean` | High-frequency vocal tract shape |
| 2 | `spectral_bandwidth_max` | Frequency spread at peak intensity |
| 3 | `mfcc_delta2_2_max` | Acceleration of cough dynamics |
| 4 | `mfcc_0_max` | Overall energy envelope |
| 5 | `mfcc_5_std` | Mid-frequency variability |

![SHAP importance](outputs/shap_importance.png)

---

## Ablation study

| Feature group | Features | AUROC |
|---------------|----------|-------|
| All features | 190 | **0.6187** |
| MFCCs only | 117 | 0.6015 |
| No MFCCs | 73 | 0.5996 |
| Spectral only | 30 | 0.5907 |
| Chroma only | 36 | 0.5502 |
| Temporal only | 6 | 0.5424 |

MFCCs are the single strongest group but the full feature set outperforms any individual group, confirming each feature type contributes independent signal.

![Ablation study](outputs/ablation_study.png)

---

## Confusion matrix (holdout)

|  | Predicted Healthy | Predicted COVID |
|--|-------------------|-----------------|
| **Actual Healthy** | 2028 | 213 |
| **Actual COVID** | 156 | 35 |

The model achieves 90% recall on healthy cases but only 18% on COVID-19 — a known limitation of classical ML on this noisy, imbalanced dataset.

---

## Setup

```bash
git clone https://github.com/jovishacurlie22/CoughDetector
cd CoughDetector
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download COUGHVID from [Zenodo](https://zenodo.org/record/7024894) and place audio files in `data/coughvid/audio/`.

Then run notebooks 01 → 02 → 03 → 04 in order.

---

## Run the demo locally

```bash
cd demo
python app.py
# Open http://127.0.0.1:7860
```

---

## Stack

| Component | Tools |
|-----------|-------|
| Audio loading | librosa, soundfile |
| Feature extraction | librosa (MFCCs, spectral, chroma) |
| Modelling | scikit-learn, XGBoost, LightGBM |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP (TreeExplainer) |
| Demo | Gradio |
| Deployment | Hugging Face Spaces |

---

## Disclaimer

⚠️ **This is a research project, not a medical diagnostic tool.** Do not use model predictions for clinical decision-making. COVID-19 diagnosis requires PCR testing by a qualified medical professional.

---

## References

## References

- Orlandic, L. et al. (2021). The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. *Scientific Data*, 8, 156. https://doi.org/10.1038/s41597-021-00937-4
- Laguarta, J. et al. (2020). COVID-19 Artificial Intelligence Diagnosis Using Only Cough Recordings. *IEEE Open Journal of Engineering in Medicine and Biology*, 1, 275–281.
- Brown, C. et al. (2020). Exploring Automatic Diagnosis of COVID-19 from Crowdsourced Respiratory Sound Data. *KDD 2020*.
