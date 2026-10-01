from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_TRANSACTION = {
    "Time": 406,
    "V1": -1.36,
    "V2": -0.07,
    "V3": 2.54,
    "V4": 1.38,
    "V5": -0.34,
    "V6": 0.46,
    "V7": -0.09,
    "V8": 0.17,
    "V9": 0.26,
    "V10": -0.53,
    "V11": 0.79,
    "V12": -0.76,
    "V13": 0.43,
    "V14": -0.90,
    "V15": 0.17,
    "V16": 0.28,
    "V17": -0.04,
    "V18": -0.18,
    "V19": 0.04,
    "V20": 0.25,
    "V21": -0.02,
    "V22": 0.27,
    "V23": -0.11,
    "V24": 0.06,
    "V25": 0.14,
    "V26": 0.10,
    "V27": 0.02,
    "V28": 0.01,
    "Amount": 10.0,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "Fraud Detection Engine"
    assert data["status"] == "running"


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "prediction" in data
    assert "decision" in data
    assert "threshold" in data


def test_prediction_probability_range():
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    data = response.json()

    assert 0 <= data["fraud_probability"] <= 1


def test_prediction_is_binary():
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    data = response.json()

    assert data["prediction"] in [0, 1]


def test_missing_feature():
    invalid_transaction = VALID_TRANSACTION.copy()

    del invalid_transaction["V17"]

    response = client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422
    