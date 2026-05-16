import pandas as pd
from sklearn.datasets import make_classification
import os

def generate_data(output_path="data/raw_data.csv"):
    # Убедимся, что папка data существует
    os.makedirs("data", exist_ok=True)
    
    print("Генерация синтетических данных...")
    # Генерируем данные для задачи бинарной классификации
    X, y = make_classification(
        n_samples=2000, 
        n_features=15, 
        n_informative=8, 
        random_state=42
    )
    
    # Упаковываем в DataFrame для удобства
    feature_cols = [f"feature_{i}" for i in range(15)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    
    # Сохраняем как CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Данные успешно сохранены в {output_path}")

if __name__ == "__main__":
    generate_data()