"""
Customer Churn Prediction - ANN Training Script
Uses scikit-learn MLPClassifier (no TensorFlow required)
Run this script first to train and save the model.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("Churn_Modelling.csv")

# Drop columns not used for prediction
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# ── 2. Encode Categorical Features ───────────────────────────────────────────

# Label-encode Gender (Male → 1, Female → 0)
label_encoder_gender = LabelEncoder()
df["Gender"] = label_encoder_gender.fit_transform(df["Gender"])

# One-hot encode Geography
onehot_encoder_geo = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
geo_encoded = onehot_encoder_geo.fit_transform(df[["Geography"]])
geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

df = df.drop(columns=["Geography"])
df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

# ── 3. Train / Test Split ────────────────────────────────────────────────────
X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Feature Scaling ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Build & Train ANN ─────────────────────────────────────────────────────
# Architecture mirrors a typical Keras ANN:
#   Input → Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
#
# MLPClassifier parameters:
#   hidden_layer_sizes : neurons per hidden layer (tuple)
#   activation         : 'relu' for hidden layers (output uses logistic/sigmoid automatically)
#   solver             : 'adam' — same optimiser as the original
#   max_iter           : epochs equivalent
#   early_stopping     : stops when val loss stops improving (like EarlyStopping callback)

print("Training ANN (MLPClassifier)...")
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # Two hidden layers: 64 → 32 neurons
    activation="relu",
    solver="adam",
    alpha=0.001,                   # L2 regularisation (weight decay)
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True,
)

model.fit(X_train_scaled, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n── Evaluation Results ──────────────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 7. Save Artefacts ────────────────────────────────────────────────────────
print("\nSaving model and encoders...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)

with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(onehot_encoder_geo, f)

print("Done! Saved: model.pkl, scaler.pkl, label_encoder_gender.pkl, onehot_encoder_geo.pkl")
print("Now run:  streamlit run app.py")
