# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("customer_churn.csv")  # Update path if needed

# Initial cleaning (handle missing values)
df.replace(" ", np.nan, inplace=True)
df.dropna(inplace=True)

# Encode categorical features
cat_cols = df.select_dtypes(include="object").columns.tolist()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (top 10 features)
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate models
best_auc = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{name} - ROC AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name
        best_y_prob = y_prob

# Confusion matrix and ROC for best model
y_pred_best = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_best)

print("\nBest Model:", best_model_name)
print("Confusion Matrix:")
print(conf_matrix)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, best_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {best_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
import joblib
joblib.dump(best_model, "best_churn_model.pkl")
