import pandas as pd
from sklearn.datasets import make_classification
import os

def generate_data(output_path="data/raw_data.csv"):
    os.makedirs("data", exist_ok=True)
    
    print("Generate data")
    X, y = make_classification(
        n_samples=2000, 
        n_features=15, 
        n_informative=8, 
        random_state=42
    )
    
    feature_cols = [f"feature_{i}" for i in range(15)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    
    df.to_csv(output_path, index=False)
    print(f"Save to {output_path}")

if __name__ == "__main__":
    generate_data()
