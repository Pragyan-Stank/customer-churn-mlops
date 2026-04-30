# src/data/ingestion.py

import pandas as pd
import os
import yaml


def load_config():
    with open("src/config/params.yaml", "r") as f:
        return yaml.safe_load(f)


def ingest_data():
    config = load_config()

    raw_path = config["data"]["raw_path"]
    output_path = config["data"]["ingested_path"]

    print(f"Reading raw data from: {raw_path}")

    df = pd.read_csv(raw_path)

    # Drop unnecessary columns
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Ingested data saved to: {output_path}")


if __name__ == "__main__":
    ingest_data()