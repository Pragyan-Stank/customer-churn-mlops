# 📊 End-to-End Customer Churn MLOps Pipeline

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DVC](https://img.shields.io/badge/DVC-reproducible-red.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking%20%26%20registry-brightgreen.svg)](https://mlflow.org/)
[![AWS ECS](https://img.shields.io/badge/AWS-ECS%20%26%20ECR%20Deployed-orange.svg)](https://aws.amazon.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container-blue.svg)](https://www.docker.com/)

A production-grade, highly-automated machine learning operations (MLOps) pipeline for predicting customer churn. The project implements a complete data/model version control workflow, hyperparameter optimization, automated testing/validation gating, local containerized orchestration, and deployment using AWS ECR and AWS ECS.

---

## 🏗️ Architecture & Pipeline Workflow

This repository orchestrates a robust MLOps lifecycle from raw data ingestion to highly available microservices in the cloud.

### 🔄 MLOps Pipeline Lifecycle (DVC + MLflow)

```text
Raw Data (Churn_Modelling.csv)
   │
   ▼
[Ingection Stage] (ingestion.py)
   │  └─ Drops identifier columns: 'RowNumber', 'CustomerId', 'Surname'
   ▼
[Preprocessing Stage] (preprocessing.py)
   │  ├─ One-hot encodes 'Geography' and 'Gender' columns
   │  ├─ Performs train-test splitting (80% / 20%)
   │  ├─ Fits StandardScaler and exports 'model/scaler.pkl' for inference
   │  └─ Saves split datasets as 'train.pkl' and 'test.pkl'
   ▼
[Training Stage] (train.py)
   │  ├─ Executes Optuna parameter study (20 trials) maximizing validation AUC
   │  ├─ Automatically tracks and logs nested trial parameters & metrics to MLflow
   │  └─ Saves metadata and best trial details to 'artifacts/train_output.json'
   ▼
[Evaluation Gating] (evaluate.py)
   │  └─ Decision check: Verifies if validation ROC-AUC >= 0.80 threshold
   ▼
[Registration Stage] (register.py)
   │  └─ If gate passes, registers the model and sets alias to 'champion' in MLflow
   ▼
[Export Stage] (export_model.py)
      └─ Fetches 'champion' model version from registry and saves local 'churn_model.pkl'
```

---

### ☁️ AWS ECS Production Architecture

The frontend static files (served by Nginx) and FastAPI inference backend are fully containerized, pushed to **Amazon ECR**, and deployed as active, scalable tasks inside an **Amazon ECS** Fargate Cluster:

```text
       [ Browser Client ]
               │
               │ (HTTP Port 80/443)
               ▼
   [ AWS Application Load Balancer / Nginx ]
               │
               ├─── (Proxy /api/* requests to Port 8000) ──► [ FastAPI Backend Task ]
               │                                               ├─ Loads 'model/scaler.pkl'
               │                                               └─ Loads 'model/churn_model.pkl'
               │
               └─── (Serve Static index.html/CSS/JS) ─────► [ Nginx Frontend Task ]
```

---

## 🛠️ Pipeline Stages Detail (DVC Lifecycle)

The machine learning lifecycle is strictly structured using DVC pipeline steps in [dvc.yaml](file:///c:/Neutron/Deep_Learning/mlops_deployment/customer-churn-mlops/dvc.yaml) and configured via [params.yaml](file:///c:/Neutron/Deep_Learning/mlops_deployment/customer-churn-mlops/src/config/params.yaml):

| Stage | Script | Purpose | Key Inputs | Key Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Ingest** | `ingestion.py` | Imports raw customer data, drops non-informative metadata identifiers (`RowNumber`, `CustomerId`, `Surname`), and outputs cleaner structured data. | `data/raw/Churn_Modelling.csv` | `data/interim/ingested.csv` |
| **Preprocess** | `preprocessing.py` | One-hot encodes categorical parameters (`Geography`, `Gender`), performs train-test splitting, standardizes features, and exports the fitted scaler artifact. | `data/interim/ingested.csv` | `data/processed/train.pkl`, `test.pkl`, `model/scaler.pkl` |
| **Train** | `train.py` | Spawns an **Optuna** hyperparameter sweep (20 trials) evaluating XGBoost parameter spaces. Logs metadata/artifacts dynamically inside nested MLflow runs. | `data/processed/train.pkl`, `test.pkl` | `artifacts/train_output.json` |
| **Evaluate** | `evaluate.py` | Assesses the best-performing trial configuration validation ROC-AUC score against a rigorous target metric (defined at `threshold_auc: 0.80`). | `artifacts/train_output.json` | `artifacts/eval_output.json` |
| **Register** | `register.py` | Checks the evaluation decision outcome. If evaluation passes, automatically registers the candidate model into the MLflow model registry and updates its alias to `champion`. | `artifacts/eval_output.json` | MLflow Registered Registry Update |
| **Export Model**| `export_model.py` | Downloads the `champion` XGBoost model directly from MLflow and exports it locally as `churn_model.pkl` along with execution tracking metadata. | MLflow Model Registry | `model/churn_model.pkl`, `model/metadata.json` |

---

## 🐳 Container Setup & Local Orchestration

The application is structured into isolated microservices configured for seamless local development and staging orchestration via `docker-compose.yml`:

### 🧬 Service Mapping

1. **Frontend (`churn-frontend`)**: Runs an Nginx web server configured as a reverse proxy. It serves the static Single Page Application (HTML/Vanilla CSS/JavaScript) and proxies all `/api/*` requests down to the backend FastAPI container using internal Docker network DNS.
2. **Backend (`churn-backend`)**: A production-ready FastAPI service. It loads `model/scaler.pkl` and `model/churn_model.pkl` on start and exposes standardized prediction schemas under `/predict`.
3. **MLflow Registry (`churn-mlflow`)**: Exposes the MLflow server at port `5000` with local sqlite storage and persistent folder mounts.

> [!NOTE]
> Backend service ports are isolated inside the internal `churn-net` network to enforce access solely through the Nginx reverse proxy, mirroring secure production topologies.

### 🚀 Running Locally

Build, instantiate, and execute the entire multi-container service stack using standard commands:

```bash
# Build the containers and launch services in detached background mode
docker-compose up --build -d

# Verify all services are online and passing checks
docker-compose ps
```

Once online, open the following services locally:
* **Frontend Web Application**: [http://localhost:3000](http://localhost:3000)
* **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Internal-only unless port exposing is customized)
* **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)

---

## ☁️ AWS ECR & ECS Production Deployment

The container images are pushed to **Amazon ECR** and deployed as scalable microservices using **Amazon ECS** with Fargate launch types.

### 📦 1. Push Container Images to AWS ECR

Execute the following commands to log into AWS, create repositories (if not already done), build production-optimized containers, and push them to ECR:

```bash
# 1. Authenticate Docker with Amazon ECR
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com

# 2. Create Amazon ECR Repositories (Only required once)
aws ecr create-repository --repository-name customer-churn-backend
aws ecr create-repository --repository-name customer-churn-frontend

# 3. Build & Tag the Backend Container
docker build -t customer-churn-backend:latest -f Dockerfile .
docker tag customer-churn-backend:latest <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/customer-churn-backend:latest

# 4. Build & Tag the Frontend Container
docker build -t customer-churn-frontend:latest -f frontend/Dockerfile ./frontend
docker tag customer-churn-frontend:latest <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/customer-churn-frontend:latest

# 5. Push both images to Amazon ECR
docker push <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/customer-churn-backend:latest
docker push <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/customer-churn-frontend:latest
```

---

### ⚙️ 2. Deploy Services inside AWS ECS (Fargate)

Deploying to AWS ECS involves configuring task definitions, networking protocols, and starting active services.

#### 🎛️ Task Definitions Layout
Create two task definitions under your ECS cluster—one for the FastAPI backend and one for the Nginx static frontend:

* **Backend Task Definition**:
  * **Launch Type**: Fargate (Serverless CPU & Memory execution)
  * **Memory**: `0.5 GB` | **CPU**: `0.25 vCPU`
  * **Image URI**: `<aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/customer-churn-backend:latest`
  * **Port Mapping**: Container port `8000` (TCP)
  * **Environment Variables**:
    * `MODEL_PATH=model/churn_model.pkl`
    * `SCALER_PATH=model/scaler.pkl`
    * `CORS_ORIGINS=*`

* **Frontend Task Definition**:
  * **Launch Type**: Fargate
  * **Memory**: `0.5 GB` | **CPU**: `0.25 vCPU`
  * **Image URI**: `<aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/customer-churn-frontend:latest`
  * **Port Mapping**: Container port `80` (TCP)

---

#### 🔀 Nginx Service Discovery & Load Balancing in ECS

When deploying multi-container services into ECS, dynamic service discovery must be preserved so the frontend knows exactly where to route `/api/*` calls:

> [!IMPORTANT]
> **Production Service Mapping**:
> In the local `docker-compose.yml`, Nginx resolves `backend:8000` using the Docker DNS. In AWS ECS, you configure this using one of two options:
>
> 1. **AWS ECS Service Connect (Recommended)**: Enable Service Connect on both services in the cluster. Register the backend service under the discovery name `backend` with port `8000` inside your private namespace. Nginx will automatically resolve `backend:8000` securely within the VPC without modifying `nginx.conf`!
> 2. **Application Load Balancer (ALB) Routing**: Configure an ALB with two listener rules:
>    * All requests to path `/api/*` are forwarded to the **Backend Target Group** (on port `8000`).
>    * All other requests (`/*`) are forwarded to the **Frontend Target Group** (on port `80`).
>    * Update `frontend/nginx.conf` upstream blocks to route to the private IP/DNS assigned by ECS.

---

## 💻 Step-by-Step Local Pipeline Execution

If you wish to re-train the model, run hyperparameter sweeps, or regenerate local model artifacts, run the local DVC pipeline:

### 1. Pre-requisites & Setup
Ensure Python 3.10 and Virtual Environment tools are set up on your machine:

```bash
# Initialize clean virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # macOS/Linux

# Install development and tracking dependencies
pip install -r requirements.txt
```

### 2. Run MLflow Server Locally
Before running training commands, boot up the local MLflow server to track experimental models:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### 3. Run Pipeline Stages
Trigger DVC execution to build or fetch stages in order:

```bash
# Execute/reproduce the complete ML lifecycle defined in dvc.yaml
dvc repro
```

---

## 🎯 Model Features Schema

The pre-trained XGBoost Model expects **11 input features** mapped exactly to standard customer attributes. Input vectors to `/predict` must be scaled using `model/scaler.pkl` before passing to the model.

| Feature Index | Raw Feature | Type | Description |
| :--- | :--- | :--- | :--- |
| `0` | `CreditScore` | Integer | Credit rating metric of customer |
| `1` | `Age` | Integer | Age of customer in years |
| `2` | `Tenure` | Integer | Number of years as a customer |
| `3` | `Balance` | Float | Financial balance in bank account |
| `4` | `NumOfProducts` | Integer | Number of products purchased |
| `5` | `HasCrCard` | Binary (`0`/`1`) | Whether the customer possesses a credit card |
| `6` | `IsActiveMember` | Binary (`0`/`1`) | Activity status of the membership profile |
| `7` | `EstimatedSalary` | Float | Estimated income of the customer |
| `8` | `Geography_Germany`| Binary (`0`/`1`) | One-hot encoded flag indicating residency in Germany |
| `9` | `Geography_Spain` | Binary (`0`/`1`) | One-hot encoded flag indicating residency in Spain |
| `10` | `Gender_Male` | Binary (`0`/`1`) | One-hot encoded flag indicating gender is Male |

---

## ✨ Key Features of the Pipeline

* **Optuna Integration**: Automatic hyperparameter sweep with validation ROC-AUC maximization.
* **Rigorous Gatekeeping**: Model validation stage prevents degradation by only registering models that pass `0.80 AUC` standards.
* **MLflow Champion Tagging**: Promoted models receive the `champion` alias tag, making programmatic extraction deterministic.
* **Nginx Reverse Proxy**: Shields the Python application server from direct public exposure, handles static assets caching, and drops invalid requests early.
* **Production Docker Build**: Multi-stage docker builds utilizing lightweight Linux bases, with non-root security configurations (`appuser`).
