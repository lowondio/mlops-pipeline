import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import joblib
import os

# mlflow ui --backend-store-uri sqlite:///mlruns.db

def train_and_log():
    print("Загрузка данных...")
    df = pd.read_csv("data/raw_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Настраиваем MLflow: указываем локальную SQLite базу для хранения логов
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # Создаем или выбираем эксперимент
    mlflow.set_experiment("my_first_pipeline")

    # 2. Запускаем трекинг
    with mlflow.start_run():
        max_iter = 150
        learning_rate = 0.1

        print("Обучение модели HistGradientBoosting...")
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, 
            learning_rate=learning_rate, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Считаем метрику
        accuracy = model.score(X_test, y_test)
        print(f"🎯 Accuracy: {accuracy:.4f}")
        
        # 3. Сохранение обученной модели (сериализация)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Модель сохранена как артефакт в {model_path}")

        # 3. Логируем данные в MLflow!
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("accuracy", accuracy)
        
        # MLflow сам сериализует модель и сохранит её версии
        mlflow.sklearn.log_model(model, "model")
        print("✅ Эксперимент и модель успешно сохранены в MLflow!")

if __name__ == "__main__":
    train_and_log()