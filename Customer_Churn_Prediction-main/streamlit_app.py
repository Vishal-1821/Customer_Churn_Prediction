import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load pre-trained model (assumed already trained on selected features)
model = joblib.load("best_churn_model.pkl")

st.title("📊 Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Initialize these as None so accessible globally
scaler = None
selector = None
selected_features = None
le_churn = None
X = None  # To keep all features for manual input reconstruction

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # Clean missing values (replace spaces with NaN)
    data.replace(" ", np.nan, inplace=True)
    data.dropna(inplace=True)

    # Encode categorical columns
    cat_cols = data.select_dtypes(include="object").columns.tolist()

    # Encode 'Churn' column last, keep it separate
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    if "Churn" not in data.columns:
        st.error("Dataset must contain 'Churn' column.")
    else:
        X = data.drop("Churn", axis=1)
        y = data["Churn"]

        # Encode target label
        le_churn = LabelEncoder()
        y_encoded = le_churn.fit_transform(y)

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Select top 10 features with chi2
        selector = SelectKBest(score_func=chi2, k=10)
        X_selected = selector.fit_transform(X_scaled, y_encoded)

        # Get selected feature names (after selection)
        selected_features = X.columns[selector.get_support()]
        st.write("✅ Selected Features:", list(selected_features))

        # Predict on selected features
        y_pred = model.predict(X_selected)

        # Convert prediction labels back to original
        y_pred_labels = le_churn.inverse_transform(y_pred)

        result_df = data.copy()
        result_df["Predicted_Churn"] = y_pred_labels

        st.subheader("📋 Prediction Results")
        st.write(result_df[["Churn", "Predicted_Churn"]].head(10))

        st.subheader("📈 Model Performance")
        st.code(classification_report(y, y_pred_labels))
        st.write("Confusion Matrix:")
        st.dataframe(confusion_matrix(y, y_pred_labels))
        st.write("ROC AUC Score:", roc_auc_score(y_encoded, y_pred))

        st.download_button("Download Predictions as CSV", result_df.to_csv(index=False), "predictions.csv", "text/csv")

# Sidebar for manual input prediction
st.sidebar.title("🔍 Predict for a New Customer")

if (
    uploaded_file is not None
    and selected_features is not None
    and scaler is not None
    and selector is not None
    and le_churn is not None
    and X is not None
):
    input_data = {}
    for col in selected_features:
        # Adjust default values if needed
        input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

    if st.sidebar.button("Predict Churn"):
        # Create a full input dictionary for all original features with default 0
        full_input = {col: 0 for col in X.columns}

        # Update only selected features with user input
        for col in selected_features:
            full_input[col] = input_data[col]

        # Create DataFrame with full feature set
        input_df = pd.DataFrame([full_input])

        # Scale full feature set
        input_scaled = scaler.transform(input_df)

        # Select features
        input_selected = selector.transform(input_scaled)

        # Predict
        churn_prediction = model.predict(input_selected)[0]

        # Convert prediction label back to original string
        churn_label = le_churn.inverse_transform([churn_prediction])[0]

        st.sidebar.success(f"Prediction: {churn_label}")

else:
    st.sidebar.info("Please upload a CSV file with data first to enable manual prediction.")
