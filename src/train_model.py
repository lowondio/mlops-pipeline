import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib
import os

# mlflow ui --backend-store-uri sqlite:///mlruns.db

def train_and_log():
    print("Data load...")
    df = pd.read_csv("data/raw_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("my_first_pipeline")

    with mlflow.start_run():
        max_iter = 150
        learning_rate = 0.1

        print("HistGradientBoosting...")
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, 
            learning_rate=learning_rate, 
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        
        os.makedirs("models", exist_ok=True)
        
        # Save learned model
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        print(f"Preserved as an artifact in {model_path}")

        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(model, "model")
        print("Successfully saved in MLflow")

if __name__ == "__main__":
    train_and_log()
