# 🧠 ADHD Risk Prediction — Streamlit App

**BTech Final Year Project | Design of Robust ML Models for ADHD Risk Classification**

---

## Project Structure

```
adhd_app/
├── app.py                  ← Main Streamlit application (686 lines)
├── adhd_model.pkl          ← Tuned XGBoost classifier (130 KB)
├── preprocessor.pkl        ← Scaler + LabelEncoders + metadata (1.2 KB)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```
Opens at → **http://localhost:8501**

### 3. Google Colab (cloud)
```python
# In a Colab cell:
!pip install streamlit pyngrok -q
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## Features

| Feature | Description |
|---|---|
| 📋 30-Question Questionnaire | 10 Attention + 10 Hyperactivity + 10 Impulsivity |
| 🎯 Auto-scoring | Attention, Hyperactivity, Impulsivity scores (0–40 each) |
| 🔍 ML Prediction | XGBoost (97% accuracy, 0.99 AUC) |
| 📊 Feature Importance | Top 15 features by XGBoost gain |
| 🔬 SHAP Explanation | Per-sample SHAP waterfall |
| 🕸️ Radar Chart | 6-domain clinical profile |
| ⚠️ Risk Gauge | Plotly speedometer with 3 risk zones |
| 💾 Full Feature Vector | All 15 engineered features shown |

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost |
| n_estimators | 1000 |
| max_depth | 5 |
| learning_rate | 0.05 |
| subsample | 0.9 |
| colsample_bytree | 0.7 |
| Test Accuracy | 97% |
| ROC-AUC | 0.9938 |
| Decision threshold | 0.25 |

**Risk Levels:**
- 🟢 **Low Risk** — probability < 40%
- 🟡 **Moderate Risk** — probability 40–70%
- 🔴 **High Risk** — probability > 70%

---

## Disclaimer
This tool is for **research and educational use only**.
A positive screening result does not constitute a diagnosis.
Consult a licensed clinician for formal DSM-5 evaluation.
