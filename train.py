"""
ChurnSense — Training Script
Trains 4 models, handles class imbalance, computes SHAP importance.
Run once locally: python train.py
On Streamlit Cloud this is called automatically from app.py.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from xgboost import XGBClassifier
import shap


# ── 1. Load & preprocess ──────────────────────────────────────────────────────
print("Loading data...")
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

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 2. Class imbalance ────────────────────────────────────────────────────────
# Dataset is ~80/20 (no churn / churn). We handle this per-model:
#   LR & RF  → class_weight='balanced'
#   XGBoost  → scale_pos_weight = majority / minority count
#   MLP      → sample_weight passed to fit()
neg, pos     = np.bincount(y_train)
scale_pw     = neg / pos                                   # ~4.0
sample_w     = compute_sample_weight("balanced", y_train)

print(f"Class balance — No churn: {neg}  Churn: {pos}  Ratio: {scale_pw:.2f}x")

# ── 3. Define & train all models ──────────────────────────────────────────────
print("\nTraining 4 models...")

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=8, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, scale_pos_weight=scale_pw,
        max_depth=5, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
        verbosity=0,
    ),
    "ANN (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        solver="adam", alpha=0.001, batch_size=32,
        learning_rate_init=0.001, max_iter=200,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42,
    ),
}

results      = {}
trained      = {}

for name, m in models.items():
    print(f"  → {name}")
    if name == "ANN (MLP)":
        m.fit(X_train_s, y_train, sample_weight=sample_w)
    else:
        m.fit(X_train_s, y_train)

    y_pred = m.predict(X_test_s)
    y_prob = m.predict_proba(X_test_s)[:, 1]

    results[name] = {
        "Accuracy (%)":  round(accuracy_score(y_test, y_pred)          * 100, 2),
        "AUC-ROC (%)":   round(roc_auc_score(y_test, y_prob)           * 100, 2),
        "Precision (%)": round(precision_score(y_test, y_pred)         * 100, 2),
        "Recall (%)":    round(recall_score(y_test, y_pred)            * 100, 2),
        "F1 Score (%)":  round(f1_score(y_test, y_pred)               * 100, 2),
    }
    trained[name] = m

# ── 4. Pick best model by AUC-ROC ─────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["AUC-ROC (%)"])
best_model = trained[best_name]
print(f"\nBest model: {best_name}  (AUC {results[best_name]['AUC-ROC (%)']}%)")

# ── 5. SHAP feature importance ────────────────────────────────────────────────
print("Computing SHAP values...")
background = X_train_s[:100]   # small background sample

if best_name in ("XGBoost", "Random Forest"):
    explainer  = shap.TreeExplainer(best_model)
    sv         = explainer.shap_values(X_test_s[:300])
    if isinstance(sv, list):    # RF returns list [class0, class1]
        sv = sv[1]
else:
    explainer  = shap.KernelExplainer(best_model.predict_proba, background)
    sv         = explainer.shap_values(X_test_s[:80])[1]

global_shap = dict(zip(feature_names, np.abs(sv).mean(axis=0).tolist()))

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
print("Saving artefacts...")
saves = {
    "model.pkl":                best_model,
    "scaler.pkl":               scaler,
    "label_encoder_gender.pkl": le_gender,
    "onehot_encoder_geo.pkl":   ohe_geo,
    "comparison.pkl":           results,
    "best_model_name.pkl":      best_name,
    "shap_global.pkl":          global_shap,
    "feature_names.pkl":        feature_names,
    "shap_background.pkl":      background,
}
for fname, obj in saves.items():
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

print("\n── Results ─────────────────────────────────────────────")
for name, r in results.items():
    tag = "  ← BEST" if name == best_name else ""
    print(f"  {name:<22} AUC {r['AUC-ROC (%)']:>5}%  F1 {r['F1 Score (%)']:>5}%{tag}")

print("\nDone. Run:  streamlit run app.py")
