import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os

# Path dataset
data_path = "MLProject/telco_churn_clean.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {os.path.abspath(data_path)}")

# Load dataset
df = pd.read_csv(data_path)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MLflow setup
mlflow.set_tracking_uri("file:MLProject/mlruns")  # simpan artefak di folder lokal mlruns
mlflow.set_experiment("Telco_Churn_Model_Balanced")
mlflow.sklearn.autolog()

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Training
with mlflow.start_run(run_name="RandomForest_Manual_Balanced"):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Metrics manual (opsional, bisa pakai autolog)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# Info lokasi artefak
mlruns_path = os.path.abspath("MLProject/mlruns")
print(f"Training selesai. Artefak MLflow disimpan di: {mlruns_path}")
