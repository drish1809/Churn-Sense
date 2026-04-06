"""
ChurnSense — Streamlit App
3 tabs: Predict · Compare Models · Feature Importance (SHAP)
Auto-trains on first boot — no subprocess.
"""

import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import shap


# ── Auto-train ─────────────────────────────────────────────────────────────────
def train_and_save():
    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])

    ohe_geo = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    geo_enc = ohe_geo.fit_transform(df[["Geography"]])
    geo_df  = pd.DataFrame(geo_enc, columns=ohe_geo.get_feature_names_out(["Geography"]))
    df = pd.concat([df.drop(columns=["Geography"]).reset_index(drop=True), geo_df], axis=1)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    neg, pos   = np.bincount(y_train)
    scale_pw   = neg / pos
    sample_w   = compute_sample_weight("balanced", y_train)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42, verbosity=0),
        "ANN (MLP)":           MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.001, batch_size=32, learning_rate_init=0.001, max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42),
    }

    results, trained = {}, {}
    for name, m in models.items():
        m.fit(X_train_s, y_train, **{"sample_weight": sample_w} if name == "ANN (MLP)" else {})
        y_pred = m.predict(X_test_s)
        y_prob = m.predict_proba(X_test_s)[:, 1]
        results[name] = {
            "Accuracy (%)":  round(accuracy_score(y_test, y_pred)  * 100, 2),
            "AUC-ROC (%)":   round(roc_auc_score(y_test, y_prob)   * 100, 2),
            "Precision (%)": round(precision_score(y_test, y_pred) * 100, 2),
            "Recall (%)":    round(recall_score(y_test, y_pred)    * 100, 2),
            "F1 Score (%)":  round(f1_score(y_test, y_pred)        * 100, 2),
        }
        trained[name] = m

    best_name  = max(results, key=lambda k: results[k]["AUC-ROC (%)"])
    best_model = trained[best_name]
    background = X_train_s[:100]

    if best_name in ("XGBoost", "Random Forest"):
        explainer = shap.TreeExplainer(best_model)
        sv = explainer.shap_values(X_test_s[:300])
        if isinstance(sv, list): sv = sv[1]
    else:
        explainer = shap.KernelExplainer(best_model.predict_proba, background)
        sv = explainer.shap_values(X_test_s[:80])[1]

    global_shap = dict(zip(feature_names, np.abs(sv).mean(axis=0).tolist()))

    for fname, obj in {
        "model.pkl": best_model, "scaler.pkl": scaler,
        "label_encoder_gender.pkl": le_gender, "onehot_encoder_geo.pkl": ohe_geo,
        "comparison.pkl": results, "best_model_name.pkl": best_name,
        "shap_global.pkl": global_shap, "feature_names.pkl": feature_names,
        "shap_background.pkl": background,
    }.items():
        with open(fname, "wb") as f: pickle.dump(obj, f)


if not os.path.exists("model.pkl"):
    with st.spinner("First-time setup: training 4 models (~60 sec)..."):
        train_and_save()
    st.rerun()


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    def _l(p):
        with open(p, "rb") as f: return pickle.load(f)
    return (_l("model.pkl"), _l("scaler.pkl"), _l("label_encoder_gender.pkl"),
            _l("onehot_encoder_geo.pkl"), _l("comparison.pkl"),
            _l("best_model_name.pkl"), _l("shap_global.pkl"),
            _l("feature_names.pkl"), _l("shap_background.pkl"))

model, scaler, le_gender, ohe_geo, comparison, best_name, shap_global, feat_names, background = load_all()


# ── Page config & CSS ──────────────────────────────────────────────────────────
st.set_page_config(page_title="ChurnSense", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
.stApp{background:#1a1510;color:#e8dfc8}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2.5rem 3.5rem 5rem;max-width:1100px}
.topbar{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:1px solid rgba(196,158,80,.25);padding-bottom:1.2rem;margin-bottom:.5rem}
.logo{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;color:#c49e50;line-height:1}
.logo span{color:#e8dfc8}
.tag{font-size:.65rem;font-weight:500;letter-spacing:.14em;text-transform:uppercase;color:rgba(196,158,80,.6)}
.panel{background:#221e16;border:1px solid rgba(196,158,80,.14);border-radius:10px;padding:1.2rem 1.6rem 1.5rem;margin-bottom:1rem}
.panel-title{font-size:.62rem;font-weight:500;letter-spacing:.16em;text-transform:uppercase;color:#c49e50;margin-bottom:1rem;display:flex;align-items:center;gap:8px}
.panel-title::after{content:'';flex:1;height:1px;background:rgba(196,158,80,.18)}
label,.stSelectbox label,.stSlider label,.stNumberInput label{color:#9a8d72!important;font-size:.8rem!important;font-weight:400!important;letter-spacing:.02em!important}
.stSelectbox>div>div{background:#161210!important;border:1px solid rgba(196,158,80,.2)!important;border-radius:8px!important;color:#e8dfc8!important}
.stNumberInput>div>div>input{background:#161210!important;border:1px solid rgba(196,158,80,.2)!important;border-radius:8px!important;color:#e8dfc8!important}
.stButton>button[kind="primary"]{background:#c49e50!important;border:none!important;border-radius:7px!important;color:#1a1510!important;font-family:'IBM Plex Sans',sans-serif!important;font-size:.85rem!important;font-weight:500!important;letter-spacing:.07em!important;text-transform:uppercase!important;padding:.65rem 1.5rem!important}
.stButton>button[kind="primary"]:hover{background:#d9b264!important}
.verdict-wrap{border-radius:10px;padding:1.8rem 2rem;margin-top:.4rem;display:flex;align-items:center;gap:1.6rem}
.v-safe{background:#13201a;border:1px solid rgba(74,175,100,.3)}
.v-risk{background:#201510;border:1px solid rgba(210,100,60,.35)}
.v-pct{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;line-height:1}
.safe{color:#5dbf7a}.risk{color:#e06840}
.v-lbl{font-size:.92rem;font-weight:500;color:#e8dfc8;margin:.2rem 0}
.v-note{font-size:.78rem;font-weight:300;color:#7a6f5c}
.gauge-outer{height:5px;background:#2e2820;border-radius:50px;overflow:hidden;margin-top:.9rem}
.gauge-safe{height:100%;border-radius:50px;background:#5dbf7a}
.gauge-risk{height:100%;border-radius:50px;background:#e06840}
.scale{display:flex;justify-content:space-between;font-size:.6rem;color:#4a4030;margin-top:.3rem}
/* comparison table */
.ctable{width:100%;border-collapse:collapse;font-size:.84rem}
.ctable th{font-size:.65rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:#c49e50;padding:.55rem .8rem;border-bottom:1px solid rgba(196,158,80,.2);text-align:left}
.ctable td{padding:.6rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);color:#c4b898}
.ctable tr:last-child td{border-bottom:none}
.ctable tr.best-row td{background:rgba(196,158,80,.06);color:#e8dfc8;font-weight:500}
.best-badge{display:inline-block;background:rgba(196,158,80,.15);border:1px solid rgba(196,158,80,.35);color:#c49e50;font-size:.62rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;padding:.15rem .5rem;border-radius:4px;margin-left:.5rem}
.imbalance-note{background:#1c1a10;border:1px solid rgba(196,158,80,.18);border-radius:8px;padding:.9rem 1.1rem;font-size:.8rem;color:#9a8d72;margin-bottom:1.2rem;line-height:1.6}
.imbalance-note strong{color:#c49e50}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid rgba(196,158,80,.18)!important;background:transparent}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none!important;color:#6a6050!important;font-size:.8rem!important;letter-spacing:.06em!important;text-transform:uppercase!important;padding:.6rem 1.4rem!important;border-radius:0!important}
.stTabs [aria-selected="true"]{color:#c49e50!important;border-bottom:2px solid #c49e50!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.6rem!important}
hr{border-color:rgba(196,158,80,.1)!important}
.footer{text-align:center;font-size:.68rem;color:#3e3828;letter-spacing:.06em;margin-top:.5rem}
.streamlit-expanderHeader{background:#221e16!important;border-radius:8px!important;color:#7a6f5c!important;font-size:.82rem!important}
</style>
""", unsafe_allow_html=True)


# ── Top bar ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="logo">Churn<span>Sense</span></div>
  <div class="tag">4 Models · SHAP Explainability · Imbalance-Aware</div>
</div>
""", unsafe_allow_html=True)


# ── SHAP chart helper ──────────────────────────────────────────────────────────
def shap_bar_chart(shap_dict, title="", n=10, highlight=None):
    """Horizontal bar chart of SHAP importances with warm colour scheme."""
    sorted_items = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    labels = [k.replace("Geography_", "Geo: ").replace("_", " ") for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#1a1510")
    ax.set_facecolor("#221e16")

    colors = ["#c49e50" if (highlight and labels[i] == highlight) else "#7a6234"
              for i in range(len(labels))]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   height=0.55, edgecolor="none")

    ax.set_xlabel("Mean |SHAP value|", color="#6a6050", fontsize=8)
    ax.tick_params(colors="#9a8d72", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.set_tick_params(colors="#4a4030")
    ax.yaxis.set_tick_params(colors="#9a8d72")
    ax.grid(axis="x", color="#2e2820", linewidth=0.6, linestyle="--")
    if title:
        ax.set_title(title, color="#c49e50", fontsize=9,
                     fontweight="normal", pad=10, loc="left")
    plt.tight_layout(pad=1.2)
    return fig


# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predict", "📊  Compare Models", "🔍  Feature Importance"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
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

    if run:
        # Build input
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
        geo_enc_inp = ohe_geo.transform([[geography]])
        geo_df_inp  = pd.DataFrame(geo_enc_inp, columns=ohe_geo.get_feature_names_out(["Geography"]))
        input_df    = pd.concat([input_df.reset_index(drop=True), geo_df_inp], axis=1)
        input_s     = scaler.transform(input_df)

        prob     = model.predict_proba(input_s)[0][1]
        churning = prob > 0.5
        pct_str  = f"{prob * 100:.1f}%"
        gauge_w  = f"{prob * 100:.1f}%"

        st.markdown("<br>", unsafe_allow_html=True)

        if churning:
            vc, pc, gc = "v-risk", "risk", "gauge-risk"
            icon   = "⚠"
            label  = "High churn risk"
            note   = "This customer shows signs of leaving. A proactive retention offer is recommended."
        else:
            vc, pc, gc = "v-safe", "safe", "gauge-safe"
            icon   = "✓"
            label  = "Low churn risk"
            note   = "This customer is likely to stay. Relationship health looks solid."

        st.markdown(f"""
        <div class="verdict-wrap {vc}">
          <div style="font-size:2.4rem;flex-shrink:0">{icon}</div>
          <div style="flex:1">
            <div class="v-pct {pc}">{pct_str}</div>
            <div class="v-lbl">{label}</div>
            <div class="v-note">{note}</div>
            <div class="gauge-outer"><div class="{gc}" style="width:{gauge_w}"></div></div>
            <div class="scale"><span>0%</span><span>50%</span><span>100%</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-prediction SHAP ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍  Why this prediction? (SHAP explanation)"):
            try:
                if best_name in ("XGBoost", "Random Forest"):
                    import shap as _shap
                    _exp = _shap.TreeExplainer(model)
                    _sv  = _exp.shap_values(input_s)
                    if isinstance(_sv, list): _sv = _sv[1]
                    sample_shap = dict(zip(feat_names, np.abs(_sv[0]).tolist()))
                else:
                    sample_shap = shap_global   # fall back to global

                fig = shap_bar_chart(sample_shap,
                                     title=f"Top features driving this prediction  ·  model: {best_name}")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Bars show how much each feature pushed the churn probability for this specific customer.")
            except Exception:
                st.caption("SHAP explanation unavailable for this prediction.")

        with st.expander("📋  View full input summary"):
            rows = [
                ("Geography", geography), ("Credit score", credit_score),
                ("Gender", gender),       ("Balance", f"${balance:,.0f}"),
                ("Age", age),             ("Est. salary", f"${estimated_salary:,.0f}"),
                ("Tenure", f"{tenure} yr"), ("Products", num_products),
                ("Credit card", "Yes" if has_cr_card else "No"),
                ("Active member", "Yes" if is_active else "No"),
            ]
            grid_html = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid rgba(196,158,80,.08)">'
                f'<span style="color:#6a6050;font-size:.78rem">{k}</span>'
                f'<span style="color:#c4b898;font-size:.8rem;font-weight:500">{v}</span></div>'
                for k, v in rows
            )
            st.markdown(f'<div style="column-count:2;column-gap:2rem">{grid_html}</div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="imbalance-note">
      <strong>Class imbalance handling</strong> — The dataset is ~80% no-churn / 20% churn.
      Without correction, models simply predict "no churn" for everyone and get 80% accuracy while missing all churners.
      Each model uses a different correction: <strong>Logistic Regression &amp; Random Forest</strong> use
      <code>class_weight='balanced'</code> · <strong>XGBoost</strong> uses <code>scale_pos_weight</code> ·
      <strong>ANN</strong> uses <code>sample_weight</code> during training.
    </div>
    """, unsafe_allow_html=True)

    metrics = ["Accuracy (%)", "AUC-ROC (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    header_cells = "".join(f"<th>{m}</th>" for m in metrics)

    rows_html = ""
    for name, r in comparison.items():
        is_best  = name == best_name
        row_cls  = "best-row" if is_best else ""
        badge    = '<span class="best-badge">★ best</span>' if is_best else ""
        cells    = "".join(f"<td>{r[m]}</td>" for m in metrics)
        rows_html += f"<tr class='{row_cls}'><td>{name}{badge}</td>{cells}</tr>"

    st.markdown(f"""
    <table class="ctable">
      <thead><tr><th>Model</th>{header_cells}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini bar chart comparing AUC across models
    names  = list(comparison.keys())
    aucs   = [comparison[n]["AUC-ROC (%)"] for n in names]
    colors = ["#c49e50" if n == best_name else "#4a3a20" for n in names]

    fig2, ax2 = plt.subplots(figsize=(7, 2.6))
    fig2.patch.set_facecolor("#1a1510")
    ax2.set_facecolor("#221e16")
    bars = ax2.bar(names, aucs, color=colors, width=0.5, edgecolor="none")
    ax2.set_ylabel("AUC-ROC (%)", color="#6a6050", fontsize=8)
    ax2.set_ylim(min(aucs) - 3, 100)
    ax2.tick_params(colors="#9a8d72", labelsize=8)
    for spine in ax2.spines.values(): spine.set_visible(False)
    ax2.yaxis.grid(True, color="#2e2820", linewidth=0.6, linestyle="--")
    ax2.set_axisbelow(True)
    for bar, val in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val}%", ha="center", va="bottom", fontsize=8, color="#c4b898")
    ax2.set_title("AUC-ROC comparison across all models", color="#c49e50",
                  fontsize=9, pad=10, loc="left")
    plt.tight_layout(pad=1.0)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.caption(f"★ Best model by AUC-ROC: **{best_name}** — used for all predictions in the Predict tab.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — GLOBAL FEATURE IMPORTANCE (SHAP)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="imbalance-note">
      <strong>How to read SHAP values</strong> — SHAP (SHapley Additive exPlanations) measures how much each
      feature contributes to the model's output on average across the test set.
      A higher bar means that feature has more influence on whether a customer churns.
      Model: <strong>{best_name}</strong>.
    </div>
    """, unsafe_allow_html=True)

    fig3 = shap_bar_chart(shap_global,
                          title="Global feature importance — mean |SHAP value| across test set",
                          n=12)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Clean names for the insight table
    sorted_feats = sorted(shap_global.items(), key=lambda x: x[1], reverse=True)
    insight_rows = ""
    insights = {
        "Age":                "Older customers churn more — likely reflecting competitors offering better rates.",
        "NumOfProducts":      "Customers with only 1 product are far more likely to leave.",
        "IsActiveMember":     "Inactive members are a strong churn signal.",
        "Balance":            "Zero-balance accounts show higher churn risk.",
        "Geography_Germany":  "German customers churn at roughly 2× the rate of French/Spanish customers.",
        "CreditScore":        "Lower credit scores correlate with slightly higher churn.",
        "EstimatedSalary":    "Salary has relatively low predictive power for churn.",
    }
    for feat, val in sorted_feats[:7]:
        clean = feat.replace("Geography_", "Geo: ").replace("_", " ")
        tip   = insights.get(feat, "—")
        insight_rows += (
            f'<tr><td style="color:#c4b898;font-size:.82rem;padding:.55rem .8rem">{clean}</td>'
            f'<td style="color:#9a8d72;font-size:.78rem;padding:.55rem .8rem">{tip}</td></tr>'
        )

    st.markdown(f"""
    <table class="ctable" style="margin-top:1.2rem">
      <thead><tr><th>Feature</th><th>Business insight</th></tr></thead>
      <tbody>{insight_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="footer">CHURNSENSE · 4 MODELS · SHAP · SCIKIT-LEARN · XGBOOST · NO TENSORFLOW</div>',
    unsafe_allow_html=True,
)
