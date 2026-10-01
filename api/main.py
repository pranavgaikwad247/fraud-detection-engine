from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predictor import FraudPredictor


app = FastAPI(
    title="Fraud Detection Engine",
    description="Machine learning API for detecting potentially fraudulent transactions.",
    version="1.0.0",
)


predictor = FraudPredictor()


class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/")
def root():
    return {
        "project": "Fraud Detection Engine",
        "status": "running",
        "version": "1.0.0",
    }


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        result = predictor.predict(
            transaction.model_dump()
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/explain")
def explain(transaction: Transaction):
    try:
        explanation = predictor.explain(
            transaction.model_dump()
        )

        return {
            "explanation": explanation
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )