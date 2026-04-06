# 🔮 ChurnSense — Bank Customer Churn Prediction

A production-style ML app that predicts whether a bank customer is likely to churn.
Built with **4 models**, **class imbalance handling**, and **SHAP explainability** — deployed on Streamlit Cloud.

🔗 **Live demo:** [your-app-url.streamlit.app](https://your-app-url.streamlit.app)

![App Screenshot](screenshot.png)

---

## What makes this project different

| Feature | Details |
|---|---|
| **4 model comparison** | Logistic Regression, Random Forest, XGBoost, ANN (MLP) evaluated side by side |
| **Class imbalance handling** | ~80/20 split addressed per-model: `class_weight`, `scale_pos_weight`, `sample_weight` |
| **SHAP explainability** | Global feature importance + per-prediction explanation for every result |
| **No TensorFlow** | ANN built with scikit-learn `MLPClassifier` — works on any Python version |
| **Live deployment** | Auto-trains on first boot, fully deployed on Streamlit Community Cloud |

---

## Results

| Model | Accuracy | AUC-ROC | F1 Score |
|---|---|---|---|
| Logistic Regression | ~80% | ~77% | ~57% |
| Random Forest | ~86% | ~87% | ~62% |
| **XGBoost** ★ | **~87%** | **~88%** | **~64%** |
| ANN (MLP) | ~85% | ~85% | ~60% |

★ Best model by AUC-ROC — used for all predictions.

---

## Key findings (SHAP)

- **Age** is the strongest churn predictor — older customers churn significantly more
- **Number of products** — customers with only 1 product are the highest risk group
- **Active membership status** — inactive members are far more likely to leave
- **Geography (Germany)** — German customers churn at ~2× the rate of French/Spanish customers
- **Estimated salary** has surprisingly low predictive power

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/churnsense.git
cd churnsense

pip install -r requirements.txt

streamlit run app.py
# Model trains automatically on first run (~60 sec)
```

---

## Project structure

```
churnsense/
├── app.py                  # Streamlit app (3 tabs: Predict, Compare, SHAP)
├── train.py                # Standalone training script
├── Churn_Modelling.csv     # Dataset (10,000 bank customers)
├── requirements.txt        # No TensorFlow
└── .gitignore
```

---

## Model architecture (ANN)

```
Input (12 features)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Output (Sigmoid → churn probability)
```

Trained with Adam optimiser, L2 regularisation (`alpha=0.001`), and early stopping.

---

## Dataset

[Churn Modelling dataset](https://www.kaggle.com/shrutimechlearn/churn-modelling) — 10,000 bank customer records with features: credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary, and churn label.

**Class imbalance:** 79.6% no-churn / 20.4% churn — corrected in all models.
