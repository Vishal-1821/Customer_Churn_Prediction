🧠 Customer Churn Prediction Dashboard:

An interactive dashboard built with Streamlit to predict customer churn using a trained machine learning classification model. The app processes user-uploaded data, performs feature selection, scaling, and prediction, and shows key evaluation metrics and a manual input section for individual predictions.

🚀 Features:

📄 CSV Upload & Auto Cleaning

🔍 Feature Selection with Chi-Squared Test

🧪 Data Normalization using MinMaxScaler

🎯 Prediction using Pre-trained ML Model (joblib)

📊 Confusion Matrix, Classification Report & ROC AUC

🧠 Manual Input for New Predictions

📥 Download Predictions as CSV

📂 Project Structure:

customer-churn-prediction/

├── streamlit_app.py # Main Streamlit application

├── best_churn_model.pkl # Pre-trained ML model (Joblib)

├── requirements.txt # Python dependencies

└── sample_data.csv # Example input data

🛠️ Installation & Running

1.Clone the repository

git clone https://github.com/your-username/customer-churn-prediction.git

cd customer-churn-prediction

2.Install dependencies

pip install -r requirements.txt

3.Run the app

streamlit run streamlit_app.py

🧪 Demo:

Upload a CSV with customer information including the Churn column. The model will:

Clean and encode data

Scale and select top 10 features

Predict churn for each customer

Display results and metrics

You can also enter values manually in the sidebar to simulate new customer predictions.

📌 Tech Stack:

Python

Streamlit

scikit-learn

Pandas, NumPy

Joblib

📈 Model Info:

The model was trained using classification algorithm (Logistic Regression) and evaluated on features selected via chi-squared test after scaling. Label encoding is used for categorical values.

💡 Use Cases:

Telecom or SaaS businesses analyzing user churn

Marketing teams targeting at-risk customers

Data science portfolio showcase
