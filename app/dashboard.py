from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "creditcard.csv"
)


FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.1rem;
        opacity: 0.7;
        margin-top: 0.3rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 650;
        margin-top: 10px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }

    .risk-number {
        font-size: 3.5rem;
        font-weight: 700;
    }

    .risk-label {
        font-size: 1.25rem;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATA_PATH)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    return df


df = load_dataset()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ Fraud Detection Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered transaction risk analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# ANALYSIS MODE
# =========================================================

mode = st.radio(
    "Analysis Mode",
    [
        "🧪 Demo Transaction",
        "🔧 Advanced Mode",
    ],
    horizontal=True,
)


# =========================================================
# DEMO TRANSACTION MODE
# =========================================================

if mode == "🧪 Demo Transaction":

    st.markdown(
        '<div class="section-title">'
        "Analyze a Real Transaction"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Select a real transaction from the dataset "
        "and send it through the production ML pipeline."
    )

    # -----------------------------------------------------
    # TRANSACTION SELECTION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "Legitimate Transaction",
                "Fraudulent Transaction",
            ],
        )

    # Select transactions based on type

    if transaction_type == "Fraudulent Transaction":

        available = df[df["Class"] == 1]

    else:

        available = df[df["Class"] == 0]


    with col2:

        transaction_index = st.number_input(
            "Transaction Number",
            min_value=0,
            max_value=len(available) - 1,
            value=0,
            step=1,
        )


    transaction = available.iloc[
        int(transaction_index)
    ]


    # -----------------------------------------------------
    # TRANSACTION DETAILS
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        "Transaction Details"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Transaction Amount",
            f"${transaction['Amount']:.2f}",
        )

    with col2:

        st.metric(
            "Transaction Time",
            f"{transaction['Time']:.0f}",
        )


    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    st.divider()

    analyze = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True,
    )


    if analyze:

        # Build API request

        transaction_data = {
            feature: float(transaction[feature])
            for feature in FEATURES
        }


        try:

            # =================================================
            # PREDICTION REQUEST
            # =================================================

            prediction_response = requests.post(
                f"{API_URL}/predict",
                json=transaction_data,
                timeout=10,
            )


            # =================================================
            # EXPLANATION REQUEST
            # =================================================

            explanation_response = requests.post(
                f"{API_URL}/explain",
                json=transaction_data,
                timeout=10,
            )


            # =================================================
            # CHECK PREDICTION RESPONSE
            # =================================================

            if prediction_response.status_code != 200:

                st.error(
                    f"Prediction API error: "
                    f"{prediction_response.status_code}"
                )

                st.stop()


            # =================================================
            # CHECK EXPLANATION RESPONSE
            # =================================================

            if explanation_response.status_code != 200:

                st.error(
                    f"Explanation API error: "
                    f"{explanation_response.status_code}"
                )

                st.stop()


            # =================================================
            # READ RESPONSES
            # =================================================

            result = prediction_response.json()

            explanation_data = (
                explanation_response.json()
            )

            explanation = explanation_data[
                "explanation"
            ]


            probability = result[
                "fraud_probability"
            ]

            prediction = result[
                "prediction"
            ]

            threshold = result[
                "threshold"
            ]


            # =================================================
            # RISK ANALYSIS
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "Risk Analysis"
                "</div>",
                unsafe_allow_html=True,
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%",
                )


            with col2:

                st.metric(
                    "Detection Threshold",
                    f"{threshold * 100:.0f}%",
                )


            # Probability bar

            st.progress(
                probability
            )


            # Decision

            if prediction == 1:

                st.error(
                    "⚠️ POTENTIAL FRAUD"
                )

                st.caption(
                    "The model identified this "
                    "transaction as potentially fraudulent."
                )

            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )

                st.caption(
                    "The model did not identify this "
                    "transaction as potentially fraudulent."
                )


            # =================================================
            # MODEL VERIFICATION
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "Model Verification"
                "</div>",
                unsafe_allow_html=True,
            )


            actual_label = int(
                transaction["Class"]
            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "**Dataset Ground Truth**"
                )

                if actual_label == 1:

                    st.error(
                        "FRAUD"
                    )

                else:

                    st.success(
                        "LEGITIMATE"
                    )


            with col2:

                st.write(
                    "**Model Prediction**"
                )

                if prediction == 1:

                    st.error(
                        "POTENTIAL FRAUD"
                    )

                else:

                    st.success(
                        "LEGITIMATE"
                    )


            # Verification message

            if prediction == actual_label:

                st.success(
                    "✓ Prediction matches the dataset label."
                )

            else:

                st.warning(
                    "Prediction differs from the dataset label."
                )


            # =================================================
            # SHAP EXPLANATION
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                "Why did the model make this decision?"
                "</div>",
                unsafe_allow_html=True,
            )


            st.caption(
                "SHAP estimates how individual features "
                "influenced this transaction's prediction."
            )


            # -------------------------------------------------
            # SHAP FEATURES
            # -------------------------------------------------

            for item in explanation:

                feature = item[
                    "feature"
                ]

                shap_value = item[
                    "shap_value"
                ]

                direction = item[
                    "direction"
                ]


                if direction == "Toward Fraud":

                    st.write(
                        f"🔴 **{feature}** — "
                        f"+{shap_value:.4f} "
                        f"toward fraud"
                    )

                else:

                    st.write(
                        f"🟢 **{feature}** — "
                        f"{shap_value:.4f} "
                        f"toward legitimate"
                    )


        # =====================================================
        # API CONNECTION ERROR
        # =====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the Fraud Detection API."
            )

            st.info(
                "Make sure FastAPI is running on "
                "http://127.0.0.1:8000"
            )


        # =====================================================
        # API TIMEOUT
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "The prediction service timed out."
            )


# =========================================================
# ADVANCED MODE
# =========================================================

else:

    st.markdown(
        '<div class="section-title">'
        "Advanced Transaction Analysis"
        "</div>",
        unsafe_allow_html=True,
    )


    st.caption(
        "V1–V28 are anonymized PCA-derived features "
        "from the training dataset."
    )


    # -----------------------------------------------------
    # BASIC TRANSACTION INFORMATION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )


    with col2:

        time = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=406.0,
            step=1.0,
        )


    # -----------------------------------------------------
    # V FEATURES
    # -----------------------------------------------------

    st.subheader(
        "Anonymized Model Features"
    )


    features = {}


    columns = st.columns(4)


    for index in range(1, 29):

        column = columns[
            (index - 1) % 4
        ]


        with column:

            features[
                f"V{index}"
            ] = st.number_input(
                f"V{index}",
                value=0.0,
                format="%.6f",
            )


    # -----------------------------------------------------
    # ANALYZE
    # -----------------------------------------------------

    st.divider()


    if st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True,
    ):

        transaction = {
            "Time": time,
            **features,
            "Amount": amount,
        }


        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=10,
            )


            if response.status_code != 200:

                st.error(
                    f"Prediction API error: "
                    f"{response.status_code}"
                )

                st.stop()


            result = response.json()


            probability = result[
                "fraud_probability"
            ]

            prediction = result[
                "prediction"
            ]

            threshold = result[
                "threshold"
            ]


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "Analysis Result"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%",
                )


            with col2:

                st.metric(
                    "Detection Threshold",
                    f"{threshold * 100:.0f}%",
                )


            st.progress(
                probability
            )


            if prediction == 1:

                st.error(
                    "⚠️ POTENTIAL FRAUD"
                )

            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the Fraud Detection API."
            )

            st.info(
                "Make sure FastAPI is running on "
                "http://127.0.0.1:8000"
            )


        except requests.exceptions.Timeout:

            st.error(
                "The prediction service timed out."
            )