"""
ADHD Risk Prediction — Streamlit Application
=============================================
Full-stack clinical ML dashboard built with:
  Streamlit · Plotly · SHAP · XGBoost · Joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, os
warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADHD Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ADHD Risk Classification System — BTech Final Year Project"}
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0d1b2a; }
[data-testid="stSidebar"]          { background: #111e2d; border-right: 1px solid #1e3a55; }
.stApp                             { background: #0d1b2a; }

/* ── Typography ── */
h1,h2,h3,h4 { color: #dde8f2 !important; }
p, label, div { color: #b8cfe4; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1c2f45;
    border: 1px solid #253d58;
    border-radius: 10px;
    padding: 14px !important;
}
[data-testid="stMetricValue"] { color: #f0f4f8 !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #7a9bb8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#f0a500,#e08800);
    color: #0d1b2a;
    font-weight: 700;
    font-size: 16px;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg,#ffb820,#f09600); transform: translateY(-1px); }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #f0a500 !important; }

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] { background: #1c2f45 !important; border-color: #253d58 !important; }

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1e3a55;
    margin: 24px 0;
}

/* ── Risk card ── */
.risk-card {
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    border: 1.5px solid;
    margin-bottom: 16px;
}
.risk-low      { background: rgba(61,184,138,0.12); border-color: #3db88a; }
.risk-moderate { background: rgba(240,165,0,0.12);  border-color: #f0a500; }
.risk-high     { background: rgba(224,90,90,0.12);  border-color: #e05a5a; }

.risk-label    { font-size: 13px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; margin-bottom:6px; }
.risk-value    { font-size: 38px; font-weight:800; line-height:1; margin-bottom:8px; }
.risk-prob     { font-size: 15px; font-weight:500; }

/* ── Info box ── */
.info-box {
    background: rgba(59,158,237,0.08);
    border: 1px solid rgba(59,158,237,0.25);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    margin: 12px 0;
}

/* ── Score badge ── */
.score-badge {
    display:inline-block; padding:3px 12px; border-radius:12px;
    font-size:12px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
}

/* ── Sidebar labels ── */
.sidebar-section {
    font-size:11px; font-weight:700; color:#7a9bb8;
    letter-spacing:.1em; text-transform:uppercase;
    margin: 18px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a55;
}

/* ── Q row ── */
.q-row {
    background:#162436; border-radius:8px; padding:10px 14px;
    margin-bottom:8px; border:1px solid #1e3a55;
    font-size:13px; color:#b8cfe4;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
FREQ_OPTIONS  = ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Very Often (4)"]
FREQ_VALUES   = {o: i for i, o in enumerate(FREQ_OPTIONS)}

ATTENTION_QS = [
    "I fail to give close attention to details or make careless mistakes.",
    "I have difficulty sustaining attention in tasks or activities.",
    "I do not seem to listen when spoken to directly.",
    "I fail to follow through on instructions or finish tasks.",
    "I have difficulty organising tasks and activities.",
    "I avoid tasks that require sustained mental effort.",
    "I lose things necessary for tasks (keys, wallet, phone).",
    "I am easily distracted by external stimuli or unrelated thoughts.",
    "I am forgetful in daily activities or appointments.",
    "I struggle to concentrate on one task before starting another.",
]

HYPERACTIVITY_QS = [
    "I fidget with hands/feet or squirm in my seat.",
    "I leave my seat when remaining seated is expected.",
    "I run about or climb in situations where it is inappropriate.",
    "I am unable to engage in leisure activities quietly.",
    "I feel 'on the go,' as if 'driven by a motor'.",
    "I talk excessively in social or academic situations.",
    "I have difficulty sitting through meetings or lectures.",
    "I feel restless or find it hard to relax in calm environments.",
    "I seek out physical activity even when it is disruptive.",
    "I struggle to stay seated during tasks requiring extended focus.",
]

IMPULSIVITY_QS = [
    "I blurt out answers before questions have been completed.",
    "I have difficulty waiting my turn in queues or conversations.",
    "I interrupt or intrude on others (conversations, games).",
    "I make hasty decisions without considering consequences.",
    "I act on the first thought without pausing to evaluate.",
    "I find it difficult to resist doing something I should not.",
    "I make purchases or commitments impulsively and later regret them.",
    "I react emotionally before fully processing a situation.",
    "I say things without thinking that offend or surprise others.",
    "I start projects impulsively without planning them through.",
]

FEATURE_LABELS = {
    "Age":              "Age",
    "Gender":           "Gender",
    "SleepHours":       "Sleep Hours",
    "ScreenTime":       "Screen Time",
    "AcademicPerformance": "Academic Performance",
    "FamilyHistory":    "Family History",
    "AttentionScore":   "Attention Score",
    "HyperactivityScore": "Hyperactivity Score",
    "ImpulsivityScore": "Impulsivity Score",
    "ClinicalTotal":    "Clinical Total",
    "SleepDeficit":     "Sleep Deficit",
    "ScreenSleep":      "Screen/Sleep Ratio",
    "AcadAttn":         "Acad × Attention",
    "HyperImpulse":     "Hyper × Impulsivity",
    "LowAcadHighClini": "Low Acad + High Clinical",
}

# ─── Load model & preprocessor ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    base = os.path.dirname(__file__)
    model       = joblib.load(os.path.join(base, "adhd_model.pkl"))
    preprocessor = joblib.load(os.path.join(base, "preprocessor.pkl"))
    explainer   = shap.TreeExplainer(model)
    return model, preprocessor, explainer

model, preprocessor, explainer = load_artifacts()

# ─── Helper: build feature vector ─────────────────────────────────────────────
def build_features(age, gender, sleep, screen, academic,
                   family, attn, hyper, impuls):
    clinical   = attn + hyper + impuls
    sleep_def  = 8 - sleep
    screen_sl  = screen / (sleep + 0.01)
    acad_attn  = academic * attn
    hyp_imp    = hyper * impuls
    low_acad   = (1 if academic < 50 else 0) * clinical

    g_enc = preprocessor["le_gender"].transform([gender])[0]
    f_enc = preprocessor["le_family"].transform([family])[0]

    row = pd.DataFrame([{
        "Age": age, "Gender": g_enc, "SleepHours": sleep,
        "ScreenTime": screen, "AcademicPerformance": academic,
        "FamilyHistory": f_enc, "AttentionScore": attn,
        "HyperactivityScore": hyper, "ImpulsivityScore": impuls,
        "ClinicalTotal": clinical, "SleepDeficit": sleep_def,
        "ScreenSleep": screen_sl, "AcadAttn": acad_attn,
        "HyperImpulse": hyp_imp, "LowAcadHighClini": low_acad,
    }])[preprocessor["feature_columns"]]

    scaled = preprocessor["scaler"].transform(row)
    return scaled, clinical, sleep_def, screen_sl, acad_attn, hyp_imp, low_acad

# ─── Helper: risk level ────────────────────────────────────────────────────────
def risk_level(prob):
    if prob < 0.40:  return "Low Risk",      "#3db88a", "risk-low"
    if prob < 0.70:  return "Moderate Risk", "#f0a500", "risk-moderate"
    return               "High Risk",       "#e05a5a", "risk-high"

# ─── Plotly theme defaults ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#b8cfe4", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px'>
      <div style='font-size:36px'>🧠</div>
      <div style='font-size:18px; font-weight:700; color:#dde8f2; margin:4px 0'>ADHD Predictor</div>
      <div style='font-size:11px; color:#7a9bb8; letter-spacing:.08em; text-transform:uppercase'>
        Clinical Risk Assessment
      </div>
    </div>
    <hr style='border-color:#1e3a55; margin:12px 0'>
    """, unsafe_allow_html=True)

    # ── Demographic inputs ────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>👤 Patient Profile</div>", unsafe_allow_html=True)

    age      = st.slider("Age", min_value=5,   max_value=60,  value=17, step=1)
    gender   = st.selectbox("Gender", ["Male", "Female"])
    family   = st.selectbox("Family History of ADHD", ["No", "Yes"])

    st.markdown("<div class='sidebar-section'>😴 Lifestyle</div>", unsafe_allow_html=True)
    sleep    = st.slider("Sleep Hours / night", 2.0, 12.0, 7.0, 0.5,
                          help="Average nightly sleep duration")
    screen   = st.slider("Screen Time hours / day", 0.0, 16.0, 4.0, 0.5,
                          help="Total daily screen time (phone, TV, computer)")

    st.markdown("<div class='sidebar-section'>📚 Academic</div>", unsafe_allow_html=True)
    academic = st.slider("Academic Performance (%)", 0, 100, 70, 1,
                          help="Overall academic score out of 100")

    st.markdown("<hr style='border-color:#1e3a55; margin:20px 0 12px'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#7a9bb8; line-height:1.6; text-align:center'>
      Model: XGBoost (tuned)<br>
      Accuracy: 97% · AUC: 0.9938<br>
      Trained on 500 clinical samples
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:28px; font-weight:800; color:#dde8f2; margin-bottom:2px'>
  🧠 ADHD Risk Classification Dashboard
</h1>
<p style='color:#7a9bb8; font-size:14px; margin-bottom:24px'>
  Complete the 30-question clinical questionnaire below,
  then click <strong style='color:#f0a500'>Predict ADHD Risk</strong> to generate your report.
</p>
""", unsafe_allow_html=True)

# ── Live score tiles ──────────────────────────────────────────────────────────
score_cols = st.columns(4)

# Will be updated after questionnaire
if "attn_score"  not in st.session_state: st.session_state.attn_score  = 0
if "hyper_score" not in st.session_state: st.session_state.hyper_score = 0
if "impuls_score"not in st.session_state: st.session_state.impuls_score= 0

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  QUESTIONNAIRE
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🎯 Attention (Q1–10)",
    "⚡ Hyperactivity (Q11–20)",
    "⚠️ Impulsivity (Q21–30)"
])

def q_block(questions, prefix, tab, color):
    responses = []
    with tab:
        st.markdown(f"""
        <div class='info-box'>
          Rate how often each symptom occurs over the <strong>past 6 months</strong>.
        </div>""", unsafe_allow_html=True)
        for i, text in enumerate(questions, 1):
            num = f"Q{i:02d}"
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"""
                <div class='q-row'>
                  <span style='color:#f0a500;font-weight:700;font-family:monospace;
                    font-size:11px'>{num}</span>
                  &nbsp;&nbsp;{text}
                </div>""", unsafe_allow_html=True)
            with c2:
                val = st.selectbox(
                    label=f"_{prefix}_{i}",
                    options=FREQ_OPTIONS,
                    label_visibility="collapsed",
                    key=f"{prefix}_{i}"
                )
                responses.append(FREQ_VALUES[val])
        total = sum(responses)
        pct   = int((total / 40) * 100)
        if total <= 10:  badge_col, badge_bg = "#3db88a", "rgba(61,184,138,0.15)"
        elif total <= 20: badge_col, badge_bg = "#3b9eed", "rgba(59,158,237,0.15)"
        elif total <= 30: badge_col, badge_bg = "#f0a500", "rgba(240,165,0,0.15)"
        else:             badge_col, badge_bg = "#e05a5a", "rgba(224,90,90,0.15)"
        st.markdown(f"""
        <div style='margin-top:16px; padding:14px 20px; border-radius:10px;
             background:{badge_bg}; border:1px solid {badge_col};
             display:flex; align-items:center; justify-content:space-between'>
          <span style='color:{badge_col}; font-weight:700; font-size:15px'>
            Section Score: {total} / 40
          </span>
          <span style='color:{badge_col}; font-size:13px; font-weight:600'>
            {pct}% severity
          </span>
        </div>""", unsafe_allow_html=True)
    return total

attn_score  = q_block(ATTENTION_QS,    "attn",  tab1, "#3b9eed")
hyper_score = q_block(HYPERACTIVITY_QS,"hyper", tab2, "#f0a500")
impuls_score= q_block(IMPULSIVITY_QS,  "impuls",tab3, "#e05a5a")
clinical_total = attn_score + hyper_score + impuls_score

# ── Live tiles ────────────────────────────────────────────────────────────────
with score_cols[0]:
    st.metric("🎯 Attention Score",    f"{attn_score} / 40",
              delta=f"{int(attn_score/40*100)}% severity")
with score_cols[1]:
    st.metric("⚡ Hyperactivity Score", f"{hyper_score} / 40",
              delta=f"{int(hyper_score/40*100)}% severity")
with score_cols[2]:
    st.metric("⚠️ Impulsivity Score",   f"{impuls_score} / 40",
              delta=f"{int(impuls_score/40*100)}% severity")
with score_cols[3]:
    st.metric("📊 Clinical Total",      f"{clinical_total} / 120",
              delta=f"{int(clinical_total/120*100)}% overall")

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("🔍 Predict ADHD Risk", type="primary")
with col_info:
    st.markdown("""
    <div class='info-box' style='margin-top:0'>
      <strong>What happens on Predict:</strong><br>
      The 9 raw inputs + 6 engineered features are passed through the
      saved XGBoost model. SHAP values explain which features drove the prediction.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if predict_clicked:
    with st.spinner("Running inference…"):
        X_scaled, clin, s_def, s_sl, a_at, h_im, l_ac = build_features(
            age, gender, sleep, screen, academic, family,
            attn_score, hyper_score, impuls_score
        )
        prob       = float(model.predict_proba(X_scaled)[0, 1])
        label, col, css_class = risk_level(prob)
        shap_vals  = explainer.shap_values(X_scaled)[0]
        feat_names = [FEATURE_LABELS[f] for f in preprocessor["feature_columns"]]
        feat_imp   = model.feature_importances_

    st.markdown("---")
    st.markdown("## 📋 Prediction Report")

    # ── Row 1: Risk gauge + score cards ───────────────────────────────────────
    r1c1, r1c2 = st.columns([1, 2])

    with r1c1:
        # Risk card
        st.markdown(f"""
        <div class='risk-card {css_class}'>
          <div class='risk-label' style='color:{col}'>ADHD Risk Level</div>
          <div class='risk-value' style='color:{col}'>{label}</div>
          <div class='risk-prob' style='color:{col}'>
            Probability: <strong>{prob:.1%}</strong>
          </div>
        </div>""", unsafe_allow_html=True)

        # Input summary
        st.markdown(f"""
        <div style='background:#162436; border-radius:10px; padding:14px 16px;
             border:1px solid #1e3a55; font-size:13px; line-height:1.9'>
          <div style='color:#7a9bb8; font-size:11px; font-weight:700;
               text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px'>
            Input Summary
          </div>
          <div>👤 {gender}, Age {age}</div>
          <div>😴 {sleep}h sleep · 📱 {screen}h screen</div>
          <div>📚 Academic: {academic}% · 🧬 Family Hx: {family}</div>
          <div>🎯 Attention: {attn_score}/40</div>
          <div>⚡ Hyperactivity: {hyper_score}/40</div>
          <div>⚠️ Impulsivity: {impuls_score}/40</div>
          <div style='color:#f0a500; font-weight:700; margin-top:6px'>
            📊 Clinical Total: {clinical_total}/120
          </div>
        </div>""", unsafe_allow_html=True)

    with r1c2:
        # Risk gauge (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": col}},
            delta={"reference": 25, "suffix": "%",
                   "increasing": {"color": "#e05a5a"},
                   "decreasing": {"color": "#3db88a"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#253d58", "tickfont": {"color": "#7a9bb8"}},
                "bar": {"color": col, "thickness": 0.28},
                "bgcolor": "#1c2f45",
                "bordercolor": "#253d58",
                "steps": [
                    {"range": [0,  40], "color": "rgba(61,184,138,0.2)"},
                    {"range": [40, 70], "color": "rgba(240,165,0,0.2)"},
                    {"range": [70,100], "color": "rgba(224,90,90,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.78,
                    "value": round(prob * 100, 1)
                }
            },
            title={"text": "ADHD Probability Score",
                   "font": {"size": 14, "color": "#b8cfe4"}}
        ))
        fig_gauge.update_layout(
            height=260,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
            margin=dict(l=20, r=20, t=50, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Threshold interpretation
        tiers = [
            ("0% – 40%",  "Low Risk",       "#3db88a", prob < 0.40),
            ("40% – 70%", "Moderate Risk",  "#f0a500", 0.40 <= prob < 0.70),
            ("70%+",      "High Risk",      "#e05a5a", prob >= 0.70),
        ]
        cols_t = st.columns(3)
        for ci, (rng, lbl, c, active) in enumerate(tiers):
            with cols_t[ci]:
                bg  = f"rgba({','.join(str(int(int(c[1:3],16))) for i in [0,2,4])},0.18)" \
                       if active else "rgba(255,255,255,0.03)"
                bd  = c if active else "#253d58"
                st.markdown(f"""
                <div style='text-align:center; background:{bg}; border:1.5px solid {bd};
                     border-radius:8px; padding:10px 6px'>
                  <div style='color:{c}; font-size:11px; font-weight:700;
                       text-transform:uppercase; letter-spacing:.06em'>{lbl}</div>
                  <div style='color:#7a9bb8; font-size:11px; margin-top:2px'>{rng}</div>
                  {'<div style="color:#f0a500;font-size:12px;font-weight:700;margin-top:4px">◀ Current</div>' if active else ''}
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Row 2: Feature importance + SHAP waterfall ────────────────────────────
    viz_c1, viz_c2 = st.columns(2)

    with viz_c1:
        st.markdown("### 📊 Feature Importance")
        sorted_idx = np.argsort(feat_imp)
        fig_fi = go.Figure(go.Bar(
            x=feat_imp[sorted_idx],
            y=[feat_names[i] for i in sorted_idx],
            orientation="h",
            marker=dict(
                color=feat_imp[sorted_idx],
                colorscale=[[0, "#1c2f45"], [0.5, "#3b9eed"], [1, "#f0a500"]],
                showscale=False,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5)
            ),
            text=[f"{v:.3f}" for v in feat_imp[sorted_idx]],
            textposition="outside",
            textfont=dict(size=10, color="#7a9bb8")
        ))
        fig_fi.update_layout(
            xaxis=dict(title="Importance Score", gridcolor="#1e3a55",
                       tickfont=dict(color="#7a9bb8")),
            yaxis=dict(tickfont=dict(color="#b8cfe4")),
            height=440, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with viz_c2:
        st.markdown("### 🔍 SHAP Explanation")
        shap_abs   = np.abs(shap_vals)
        shap_order = np.argsort(shap_abs)
        shap_colors = ["#e05a5a" if v > 0 else "#3db88a" for v in shap_vals[shap_order]]

        fig_shap = go.Figure(go.Bar(
            x=shap_vals[shap_order],
            y=[feat_names[i] for i in shap_order],
            orientation="h",
            marker=dict(
                color=shap_colors,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5)
            ),
            text=[f"{v:+.3f}" for v in shap_vals[shap_order]],
            textposition="outside",
            textfont=dict(size=10, color="#7a9bb8")
        ))
        fig_shap.add_vline(x=0, line_color="#253d58", line_width=1.5)
        fig_shap.update_layout(
            xaxis=dict(title="SHAP Value (impact on prediction)",
                       gridcolor="#1e3a55", tickfont=dict(color="#7a9bb8"),
                       zeroline=False),
            yaxis=dict(tickfont=dict(color="#b8cfe4")),
            height=440, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("""
        <div class='info-box'>
          <span style='color:#e05a5a;font-weight:700'>Red bars</span> push the prediction
          toward ADHD. <span style='color:#3db88a;font-weight:700'>Green bars</span>
          push it toward No-ADHD. Bar length = magnitude of influence.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Row 3: Domain score radar + score breakdown ───────────────────────────
    radar_c1, radar_c2 = st.columns(2)

    with radar_c1:
        st.markdown("### 🕸️ Clinical Domain Radar")
        categories = ["Attention", "Hyperactivity", "Impulsivity",
                      "Sleep Risk", "Screen Risk", "Academic Risk"]
        vals = [
            attn_score / 40 * 100,
            hyper_score / 40 * 100,
            impuls_score / 40 * 100,
            max(0, (8 - sleep) / 8 * 100),
            min(100, screen / 16 * 100),
            max(0, (100 - academic)),
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.18)",
            line=dict(color=col, width=2),
            marker=dict(size=6, color=col)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#1e3a55", tickfont=dict(color="#7a9bb8",size=9)),
                angularaxis=dict(gridcolor="#1e3a55", tickfont=dict(color="#b8cfe4",size=11))
            ),
            showlegend=False,
            height=360, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with radar_c2:
        st.markdown("### 📈 Score Breakdown")
        domains = ["Attention", "Hyperactivity", "Impulsivity"]
        scores  = [attn_score, hyper_score, impuls_score]
        dom_colors = ["#3b9eed", "#f0a500", "#e05a5a"]

        fig_bar = go.Figure()
        for dom, sc, dc in zip(domains, scores, dom_colors):
            fig_bar.add_trace(go.Bar(
                x=[dom], y=[sc],
                name=dom,
                marker_color=dc,
                marker_line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                text=[f"{sc}/40"],
                textposition="outside",
                textfont=dict(color="#f0f4f8", size=13)
            ))
        fig_bar.add_hline(y=20, line_dash="dash", line_color="#7a9bb8",
                          annotation_text="Moderate threshold",
                          annotation_font_color="#7a9bb8")
        fig_bar.add_hline(y=30, line_dash="dot", line_color="#e05a5a",
                          annotation_text="High threshold",
                          annotation_font_color="#e05a5a")
        fig_bar.update_layout(
            yaxis=dict(range=[0, 48], gridcolor="#1e3a55",
                       title="Score", tickfont=dict(color="#7a9bb8")),
            xaxis=dict(tickfont=dict(color="#b8cfe4")),
            showlegend=False,
            barmode="group",
            height=360, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Row 4: Full feature vector table ─────────────────────────────────────
    with st.expander("🔬 Full Feature Vector (sent to model)", expanded=False):
        feat_raw = {
            "Feature": feat_names,
            "Raw Value": [
                age, gender, sleep, screen, academic, family,
                attn_score, hyper_score, impuls_score,
                clinical_total, round(8-sleep,2),
                round(screen/(sleep+0.01),3),
                attn_score*academic, hyper_score*impuls_score,
                (1 if academic<50 else 0)*clinical_total
            ],
            "Scaled [0–1]": X_scaled[0].round(4).tolist(),
            "SHAP Value":   [f"{v:+.4f}" for v in shap_vals],
            "Importance":   feat_imp.round(4).tolist(),
        }
        df_feat = pd.DataFrame(feat_raw)
        # Round numeric columns for display (no .style — compatible with all pandas versions)
        df_feat["Scaled [0–1]"] = df_feat["Scaled [0–1]"].apply(lambda x: f"{x:.4f}")
        df_feat["Importance"]   = df_feat["Importance"].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_feat, use_container_width=True, hide_index=True)

    # ── Clinical disclaimer ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:rgba(240,165,0,0.06); border:1px solid rgba(240,165,0,0.2);
         border-radius:10px; padding:16px 20px; font-size:13px; color:#7a9bb8;
         line-height:1.6; margin-top:8px'>
      <strong style='color:#f0a500'>⚕️ Clinical Disclaimer:</strong>
      This tool is for <strong>research and educational use only</strong>.
      Result: <strong style='color:{col}'>{label}</strong>
      (p = {prob:.4f}). A positive screening does not constitute a diagnosis.
      Please consult a licensed clinician for formal evaluation per DSM-5 criteria.
    </div>""", unsafe_allow_html=True)

else:
    # ── Placeholder state ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:48px 24px; background:#162436;
         border-radius:14px; border:1px dashed #253d58; margin-top:12px'>
      <div style='font-size:48px; margin-bottom:12px'>🔍</div>
      <div style='font-size:17px; font-weight:600; color:#dde8f2; margin-bottom:6px'>
        Complete the questionnaire and click Predict
      </div>
      <div style='font-size:13px; color:#7a9bb8; max-width:400px; margin:0 auto'>
        Fill in the 30 clinical questions across the three tabs above,
        then click <strong style='color:#f0a500'>Predict ADHD Risk</strong>
        to generate your full prediction report with SHAP explanations.
      </div>
    </div>""", unsafe_allow_html=True)
