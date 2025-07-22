import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier, Pool

# === Paths ===
MODEL_PATH = "model/catboost_model.cbm"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# === Categorical columns ===
CATEGORICAL_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

# === Streamlit config ===
st.set_page_config(page_title="Churn Project")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
    return df

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

# === App Title & Description ===
st.title("📉 Telco Customer Churn Prediction")

st.markdown(
    "This app predicts the probability that a customer will churn based on their service and demographic information. "
    "It uses a trained CatBoost machine learning model to evaluate risk. "
    "You can also view top at-risk customers and interpret individual predictions using SHAP."
)

# === App Logic ===
def main():
    model = load_model()
    data = load_data()

    max_tenure = data['tenure'].max()
    max_monthly_charges = data['MonthlyCharges'].max()
    max_total_charges = data['TotalCharges'].max()

    election = st.radio("What would you like to do?", (
        "Calculate the probability of CHURN",
        "Show top K customers at risk"
    ))

    # ========== OPTION 1 ==========
    if election == "Calculate the probability of CHURN":
        customerID = "9999-TEST"
        gender = st.selectbox("Gender:", ("Female", "Male"))
        senior_citizen = st.number_input("SeniorCitizen (0 = No, 1 = Yes)", min_value=0, max_value=1)
        partner = st.selectbox("Partner:", ("No", "Yes"))
        dependents = st.selectbox("Dependents:", ("No", "Yes"))
        tenure = st.number_input("Tenure:", min_value=0, max_value=max_tenure)
        phone_service = st.selectbox("PhoneService:", ("No", "Yes"))
        multiple_lines = st.selectbox("MultipleLines:", ("No", "Yes"))
        internet_service = st.selectbox("InternetService:", ("No", "DSL", "Fiber optic"))
        online_security = st.selectbox("OnlineSecurity:", ("No", "Yes"))
        online_backup = st.selectbox("OnlineBackup:", ("No", "Yes"))
        device_protection = st.selectbox("DeviceProtection:", ("No", "Yes"))
        tech_support = st.selectbox("TechSupport:", ("No", "Yes"))
        streaming_tv = st.selectbox("StreamingTV:", ("No", "Yes"))
        streaming_movies = st.selectbox("StreamingMovies:", ("No", "Yes"))
        contract = st.selectbox("Contract:", ("Month-to-month", "One year", "Two year"))
        paperless_billing = st.selectbox("PaperlessBilling:", ("No", "Yes"))
        payment_method = st.selectbox("PaymentMethod:", (
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ))
        monthly_charges = st.number_input("Monthly Charges:", min_value=0.0, max_value=max_monthly_charges)
        total_charges = st.number_input("Total Charges:", min_value=0.0, max_value=max_total_charges)

        # === On confirm button ===
        if st.button("Confirm"):
            new_customer = pd.DataFrame({
                "customerID": [customerID],
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges]
            })

            # Ensure column order matches model input
            new_customer = new_customer[model.feature_names_]

            # CatBoost Pool for categorical handling
            cat_feature_indices = [new_customer.columns.get_loc(col) for col in CATEGORICAL_COLS]
            pool = Pool(new_customer, cat_features=cat_feature_indices)

            # Predict
            prob = model.predict_proba(pool)[:, 1]
            st.markdown(f"<h2>Churn Probability: {prob[0]:.2%}</h2>", unsafe_allow_html=True)

            st.subheader("Entered Customer Data")
            st.dataframe(new_customer)

            # === SHAP Waterfall Plot ===
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(new_customer)

            st.subheader("🔍 Why did the model predict this?")
            st.markdown("""
            **How to read this chart**:
            - Red bars push the customer toward **higher churn risk**
            - Blue bars push them toward **lower churn risk**
            - The length of the bars = impact strength
            - This customer's predicted churn probability is calculated from the sum of these effects
            """)

            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[0],
                feature_names=new_customer.columns,
                max_display=10,
                show=False
            )
            st.pyplot(fig)
            plt.savefig("shap_waterfall.png")
            plt.close()

    # ========== OPTION 2 ==========
    elif election == "Show top K customers at risk":
        K = st.slider("Select number of customers to display", min_value=1, max_value=50, value=10)

        # Pool with proper cat features
        pool = Pool(data[model.feature_names_], cat_features=CATEGORICAL_COLS)
        churn_probs = model.predict_proba(pool)[:, 1]

        # Sort top-K
        data['Churn Probability'] = churn_probs
        top_k = data.sort_values(by='Churn Probability', ascending=False).head(K)

        st.subheader(f"🔝 Top {K} Customers at Risk of Churn")
        st.dataframe(top_k[['customerID', 'Churn Probability'] + model.feature_names_])

if __name__ == "__main__":
    main()
