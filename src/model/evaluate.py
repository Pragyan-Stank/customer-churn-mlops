# src/model/evaluate.py

import json
import os
import yaml


# -------------------------------
# Load config
# -------------------------------
with open("src/config/params.yaml", "r") as f:
    config = yaml.safe_load(f)

THRESHOLD_AUC = config["evaluation"]["threshold_auc"]


# -------------------------------
# Evaluation Logic
# -------------------------------
def evaluate():

    # -------------------------------
    # Load training output
    # -------------------------------
    with open("artifacts/train_output.json", "r") as f:
        train_output = json.load(f)

    best_run_id = train_output["best_run_id"]
    best_val_auc = train_output["best_val_auc"]

    print(f"Best Run ID: {best_run_id}")
    print(f"Best Validation AUC: {best_val_auc}")

    # -------------------------------
    # Decision logic (from config)
    # -------------------------------
    if best_val_auc >= THRESHOLD_AUC:
        decision = "pass"
    else:
        decision = "fail"

    print(f"Threshold AUC: {THRESHOLD_AUC}")
    print(f"Evaluation Decision: {decision.upper()}")

    # -------------------------------
    # Save evaluation output
    # -------------------------------
    os.makedirs("artifacts", exist_ok=True)

    eval_output = {
        "best_run_id": best_run_id,
        "best_val_auc": best_val_auc,
        "threshold_auc": THRESHOLD_AUC,
        "decision": decision
    }

    with open("artifacts/eval_output.json", "w") as f:
        json.dump(eval_output, f)

    return eval_output


# -------------------------------
# Entry point (DVC compatible)
# -------------------------------
if __name__ == "__main__":
    evaluate()