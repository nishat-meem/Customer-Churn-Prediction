import pandas as pd
from catboost import CatBoostClassifier, Pool
import os



CATEGORICAL_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

model = CatBoostClassifier()
model_path = os.path.join(os.getcwd(), "model", "catboost_model.cbm")
model.load_model(model_path)

def predict_churn(data_dict: dict) -> float:
    df = pd.DataFrame([data_dict])
    df = df[model.feature_names_]
    cat_features = [df.columns.get_loc(col) for col in CATEGORICAL_COLS]
    pool = Pool(df, cat_features=cat_features)
    prob = model.predict_proba(pool)[:, 1]
    return float(prob[0])
