# 🛡️ Fraud Detection Engine

An end-to-end machine learning system for detecting potentially fraudulent credit card transactions.

The project covers the complete ML lifecycle:

**Data → Model Training → Evaluation → Explainability → API → Dashboard → Docker**

---

## 🚀 Features

- Fraud detection using a Random Forest classifier
- Imbalanced classification handling
- Validation-based fraud threshold selection
- REST API using FastAPI
- Interactive Streamlit dashboard
- SHAP-based local model explanations
- Automated API tests using pytest
- Dockerized inference service
- Docker Compose for the complete application stack
- API health check
- Reproducible scikit-learn inference environment

---

## 🏗️ Architecture

```text
                    Fraud Detection Engine
                            │
                    Docker Compose
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      Streamlit Dashboard            FastAPI API
          :8501                         :8000
             │                             │
             └──────────────► Prediction ──┤
                                           │
                                      Random Forest
                                           │
                                         SHAP
                                           │
                                    Model Artifacts
                                    