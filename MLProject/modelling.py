import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os
import argparse

# --------------------------------------
# Parsing argument untuk experiment
# --------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Nama experiment MLflow. Jika None, pakai default experiment."
)
args = parser.parse_args()

# --------------------------------------
# Setup MLflow tracking URI
# --------------------------------------
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Jika ada experiment name, set experiment, kalau tidak pakai default
if args.experiment_name:
    mlflow.set_experiment(args.experiment_name)

# Aktifkan autolog untuk scikit-learn
mlflow.sklearn.autolog()

# --------------------------------------
# Load data
# --------------------------------------
data_path = os.path.join(os.path.dirname(__file__), "telco_churn_clean.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# --------------------------------------
# Split train-test
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------
# Definisikan model
# --------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------
# Fit model dan otomatis log via MLflow
# --------------------------------------
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nTraining selesai. Buka MLflow UI untuk melihat hasil.")
