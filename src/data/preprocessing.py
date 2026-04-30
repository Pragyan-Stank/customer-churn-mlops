# src/data/preprocessing.py

import pandas as pd
import os
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config():
    with open("src/config/params.yaml", "r") as f:
        return yaml.safe_load(f)


def preprocess():
    config = load_config()

    input_path = config["data"]["ingested_path"]
    train_output = config["data"]["train_path"]
    test_output = config["data"]["test_path"]

    print(f"Loading data from: {input_path}")

    df = pd.read_csv(input_path)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    # Split features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config["training"]["random_seed"]
    )

    # Scaling
    scaler = StandardScaler()

    X_train_trf = scaler.fit_transform(X_train)
    X_test_trf = scaler.transform(X_test)

    # Save processed data
    os.makedirs(os.path.dirname(train_output), exist_ok=True)

    with open(train_output, "wb") as f:
        pickle.dump((X_train_trf, y_train), f)

    with open(test_output, "wb") as f:
        pickle.dump((X_test_trf, y_test), f)

    print("Preprocessing complete!")
    print(f"Train data saved to: {train_output}")
    print(f"Test data saved to: {test_output}")


if __name__ == "__main__":
    preprocess()