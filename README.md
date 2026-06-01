# Fraud Detection with ML

Compare **XGBoost**, **Isolation Forest**, and **Autoencoder** on synthetic transaction data. Includes SHAP explainability and a Streamlit dashboard for real-time scoring.

## Model Comparison

| Model | Recall (Fraud) | Precision (Fraud) | Notes |
|-------|:-:|:-:|-------|
| **XGBoost** (threshold=0.25) | **0.79** | 0.18 | Best performer, handles class imbalance via `scale_pos_weight` |
| Isolation Forest | 0.22 | 0.20 | Unsupervised, 3% contamination |
| Autoencoder | 0.07 | 0.06 | Reconstruction error threshold at 97th percentile |

## Project Structure

```
src/generate_data.py    — Synthetic data generator with fraud signals
notebooks/01_eda.ipynb  — Full EDA, training, SHAP analysis
dashboard/dashboard.py  — Streamlit dashboard for transaction scoring
```

## Quick Start

```bash
pip install -r requirements.txt
python src/generate_data.py     # generate 100K transactions
streamlit run dashboard/dashboard.py
```

## Fraud Signals

Transactions are generated with engineered signals: odd-hour transactions, international payments, new merchants, high amount-to-balance ratio, rapid consecutive transactions, and round-amount patterns. XGBoost achieves 0.935 ROC-AUC on the test set.
