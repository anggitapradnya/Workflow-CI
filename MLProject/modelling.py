import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="MLProject/telco_churn_clean.csv"
)
args = parser.parse_args()


df = pd.read_csv(args.data_path)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


mlflow.set_experiment("Telco_Churn_CI")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RF_CI_Run"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy_manual", acc)
    mlflow.log_metric("f1_manual", f1)
    mlflow.log_metric("precision_manual", prec)
    mlflow.log_metric("recall_manual", rec)
    mlflow.log_metric("roc_auc_manual", roc)

    print("Training selesai")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
