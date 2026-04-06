# 🔮 ChurnSense — Customer Churn Prediction

An ANN-powered Streamlit app that predicts whether a bank customer is likely to churn.  
Built with **scikit-learn MLPClassifier** — no TensorFlow required.

## Features
- Artificial Neural Network (2 hidden layers: 64 → 32 neurons, ReLU, Adam)
- Interactive Streamlit UI with a dark fintech aesthetic
- Probability gauge + colour-coded risk verdict
- Works on any Python version (no TensorFlow dependency)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churnsense.git
cd churnsense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates .pkl files)
python train.py

# 4. Run the app
streamlit run app.py
```

## Project Structure

```
churnsense/
├── Churn_Modelling.csv       # Dataset
├── train.py                  # ANN training script
├── app.py                    # Streamlit web app
├── requirements.txt          # Dependencies (no TensorFlow)
└── .gitignore
```

## Dataset
The [Churn Modelling dataset](https://www.kaggle.com/shrutimechlearn/churn-modelling) contains 10,000 bank customer records with features like credit score, geography, age, balance, and whether the customer exited (churned).

## Model Architecture

| Layer  | Neurons | Activation |
|--------|---------|------------|
| Input  | 12      | —          |
| Hidden | 64      | ReLU       |
| Hidden | 32      | ReLU       |
| Output | 1       | Sigmoid    |

Trained with Adam optimiser, L2 regularisation, and early stopping.
