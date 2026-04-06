"""
Customer Churn Prediction — Streamlit App (v2 Redesign)
Warm amber + forest green aesthetic. Run train.py first.
"""

import os
import subprocess
import streamlit as st
import pandas as pd
import pickle

# Auto-train if model files are missing (runs on first Streamlit Cloud boot)
if not os.path.exists("model.pkl"):
    with st.spinner("First-time setup: training the model, please wait..."):
        subprocess.run(["python", "train.py"], check=True)

st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #1a1510; color: #e8dfc8; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3.5rem 5rem 3.5rem; max-width: 1080px; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid rgba(196,158,80,0.25);
    padding-bottom: 1.4rem;
    margin-bottom: 2.6rem;
}
.topbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #c49e50;
    letter-spacing: -0.01em;
    line-height: 1;
}
.topbar-logo span { color: #e8dfc8; }
.topbar-tag {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(196,158,80,0.65);
}

/* ── Panel cards ── */
.panel {
    background: #221e16;
    border: 1px solid rgba(196,158,80,0.14);
    border-radius: 12px;
    padding: 1.5rem 1.7rem 1.8rem;
    margin-bottom: 1.4rem;
}
.panel-title {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #c49e50;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(196,158,80,0.18);
}

/* ── Widget label overrides ── */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #9a8d72 !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
}

/* ── Inputs ── */
.stSelectbox > div > div {
    background: #161210 !important;
    border: 1px solid rgba(196,158,80,0.2) !important;
    border-radius: 8px !important;
    color: #e8dfc8 !important;
}
.stNumberInput > div > div > input {
    background: #161210 !important;
    border: 1px solid rgba(196,158,80,0.2) !important;
    border-radius: 8px !important;
    color: #e8dfc8 !important;
}

/* ── Predict button ── */
.stButton > button[kind="primary"] {
    background: #c49e50 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #1a1510 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    transition: background 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #d9b264 !important;
}

/* ── Verdict box ── */
.verdict-wrap {
    border-radius: 12px;
    padding: 2.2rem 2.4rem;
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.verdict-safe {
    background: #13201a;
    border: 1px solid rgba(74,175,100,0.3);
}
.verdict-risk {
    background: #201510;
    border: 1px solid rgba(210,100,60,0.35);
}
.verdict-icon {
    font-size: 2.8rem;
    line-height: 1;
    flex-shrink: 0;
}
.verdict-pct {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.pct-safe  { color: #5dbf7a; }
.pct-risk  { color: #e06840; }
.verdict-label {
    font-size: 1.0rem;
    font-weight: 500;
    color: #e8dfc8;
    margin-bottom: 0.2rem;
}
.verdict-note {
    font-size: 0.82rem;
    font-weight: 300;
    color: #7a6f5c;
}

/* ── Gauge ── */
.gauge-outer {
    height: 6px;
    background: #2e2820;
    border-radius: 50px;
    overflow: hidden;
    margin-top: 1.2rem;
}
.gauge-inner-safe {
    height: 100%;
    border-radius: 50px;
    background: #5dbf7a;
}
.gauge-inner-risk {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #e08840, #e06840);
}
.gauge-scale {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #4a4030;
    margin-top: 0.35rem;
    letter-spacing: 0.04em;
}

/* ── Summary grid ── */
.sumgrid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0;
}
.sumrow {
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(196,158,80,0.08);
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}
.sumrow:nth-child(odd)  { padding-right: 1.5rem; }
.sumrow:nth-child(even) { padding-left: 1.5rem; border-left: 1px solid rgba(196,158,80,0.08); }
.sumrow-key   { font-size: 0.78rem; color: #6a6050; font-weight: 400; }
.sumrow-val   { font-size: 0.82rem; color: #c4b898; font-weight: 500; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #221e16 !important;
    border-radius: 8px !important;
    color: #7a6f5c !important;
    font-size: 0.82rem !important;
}

/* ── Divider + footer ── */
hr { border-color: rgba(196,158,80,0.1) !important; }
.footer {
    text-align: center;
    font-size: 0.74rem;
    color: #3e3828;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    with open("model.pkl",               "rb") as f: model     = pickle.load(f)
    with open("scaler.pkl",              "rb") as f: scaler    = pickle.load(f)
    with open("label_encoder_gender.pkl","rb") as f: le_gender = pickle.load(f)
    with open("onehot_encoder_geo.pkl",  "rb") as f: ohe_geo   = pickle.load(f)
    return model, scaler, le_gender, ohe_geo

try:
    model, scaler, le_gender, ohe_geo = load_artefacts()
except FileNotFoundError:
    st.error("Model files not found — run `python train.py` first.")
    st.stop()


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">Churn<span>Sense</span></div>
    <div class="topbar-tag">Neural Network · Customer Intelligence</div>
</div>
""", unsafe_allow_html=True)


# ── Input panels ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="panel"><div class="panel-title">Personal profile</div>', unsafe_allow_html=True)
    geography = st.selectbox("Country / Geography", ohe_geo.categories_[0])
    gender    = st.selectbox("Gender", le_gender.classes_)
    age       = st.slider("Age", 18, 92, 38)
    tenure    = st.slider("Years with bank", 0, 10, 4)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="panel"><div class="panel-title">Financial details</div>', unsafe_allow_html=True)
    credit_score     = st.number_input("Credit score", min_value=300, max_value=850, value=660)
    balance          = st.number_input("Account balance ($)", min_value=0.0, value=55000.0, step=500.0, format="%.2f")
    estimated_salary = st.number_input("Estimated annual salary ($)", min_value=0.0, value=62000.0, step=500.0, format="%.2f")
    num_products     = st.slider("Number of products held", 1, 4, 1)
    st.markdown('</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")
with col_a:
    has_cr_card = st.selectbox("Holds a credit card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
with col_b:
    is_active = st.selectbox("Active member?", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("RUN PREDICTION", type="primary", use_container_width=True)


# ── Prediction ─────────────────────────────────────────────────────────────────
if run:
    input_df = pd.DataFrame({
        "CreditScore":     [credit_score],
        "Gender":          [le_gender.transform([gender])[0]],
        "Age":             [age],
        "Tenure":          [tenure],
        "Balance":         [balance],
        "NumOfProducts":   [num_products],
        "HasCrCard":       [has_cr_card],
        "IsActiveMember":  [is_active],
        "EstimatedSalary": [estimated_salary],
    })
    geo_enc = ohe_geo.transform([[geography]])
    geo_df  = pd.DataFrame(geo_enc, columns=ohe_geo.get_feature_names_out(["Geography"]))
    input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

    prob     = model.predict_proba(scaler.transform(input_df))[0][1]
    churning = prob > 0.5
    pct_str  = f"{prob * 100:.1f}%"
    gauge_w  = f"{prob * 100:.1f}%"

    st.markdown("<br>", unsafe_allow_html=True)

    if churning:
        verdict_class = "verdict-risk"
        pct_class     = "pct-risk"
        gauge_class   = "gauge-inner-risk"
        icon          = "⚠"
        label         = "High churn risk"
        note          = "This customer shows signs of leaving. A proactive retention offer is recommended."
    else:
        verdict_class = "verdict-safe"
        pct_class     = "pct-safe"
        gauge_class   = "gauge-inner-safe"
        icon          = "✓"
        label         = "Low churn risk"
        note          = "This customer is likely to stay. Relationship health looks solid."

    st.markdown(f"""
    <div class="verdict-wrap {verdict_class}">
        <div class="verdict-icon">{icon}</div>
        <div style="flex:1">
            <div class="verdict-pct {pct_class}">{pct_str}</div>
            <div class="verdict-label">{label}</div>
            <div class="verdict-note">{note}</div>
            <div class="gauge-outer">
                <div class="{gauge_class}" style="width:{gauge_w}"></div>
            </div>
            <div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("View full input summary"):
        items = [
            ("Geography",       geography),
            ("Credit score",    credit_score),
            ("Gender",          gender),
            ("Balance",         f"${balance:,.0f}"),
            ("Age",             age),
            ("Est. salary",     f"${estimated_salary:,.0f}"),
            ("Tenure",          f"{tenure} yr"),
            ("Products",        num_products),
            ("Credit card",     "Yes" if has_cr_card else "No"),
            ("Active member",   "Yes" if is_active else "No"),
        ]
        rows_html = "".join(
            f'<div class="sumrow"><span class="sumrow-key">{k}</span><span class="sumrow-val">{v}</span></div>'
            for k, v in items
        )
        st.markdown(f'<div class="sumgrid">{rows_html}</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer">CHURNSENSE · ANN · SCIKIT-LEARN · NO TENSORFLOW</div>', unsafe_allow_html=True)
