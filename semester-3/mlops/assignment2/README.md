# Binary Image Classification – MLOps Pipeline (Cats vs Dogs)

Welcome to the end-to-end MLOps pipeline for a Binary Image Classification system. This project tackles a Cats vs. Dogs image classification problem tailored for a pet adoption platform, taking the lifecycle from data versioning to continuous integration, delivery, and robust production monitoring.

**Note:** This was built as an academic project to demonstrate a complete MLOps lifecycle from local to production environments.

## Important Links
- **Assignment Instructions:** [MLOPS-Assignment-2-Group-3.pdf](./MLOPS-Assignment-2-Group-3.pdf)
- **GitHub Repository:** [binary-image-classification-mlops](https://github.com/KSharma-SourceCode/binary-image-classification-mlops)
- **Docker Hub Images:** [ksharmadockerhub/binary-image-classification-mlops](https://hub.docker.com/repository/docker/ksharmadockerhub/binary-image-classification-mlops/general)
- **CI/CD Workflows:** [GitHub Actions Dashboard](https://github.com/KSharma-SourceCode/binary-image-classification-mlops/actions)
- **Video Walkthrough:** [Watch the Video Explanation](https://drive.google.com/file/d/18WJlB-jpkpWz8dW5ikWm6f63RmHeY5fu/view?usp=sharing)

---

## Project Overview
* **Objective:** Design and implement a complete MLOps pipeline to predict whether an uploaded image contains a cat or a dog, returning both the label and a confidence score.
* **Scope:** 
  - Data & code versioning (Git + DVC)
  - Model development and experiment tracking (MLflow)
  - Inference service exposing API (FastAPI)
  - Containerization (Docker)
  - Automated CI pipelines (GitHub Actions)
  - Continuous Deployment / delivery flow
  - Live system monitoring and telemetry (Prometheus)

---

## Group Details (Group 3)
| S. No | Name | BITS ID | Contribution |
| :--- | :--- | :--- | :--- |
| 1 | Karan Sharma | 2024AB05145 | 100% |
| 2 | Selva Pandian | 2023AC05005 | 100% |
| 3 | Neerumalla Kavitha | 2024AA05879 | 100% |
| 4 | Suraj Anand | 2024AA05731 | 100% |
| 5 | Shikhar Nigam | 2024AA05691 | 100% |

---

## High-Level Architecture
1. **Source & Version Control:** Git manages code history and lightweight metadata (e.g., `.dvc` files), while **DVC** versions large assets like image datasets and trained models.
2. **Model Training & Tracking:** A Baseline CNN (TensorFlow/Keras) trains on the dataset (224x224 RGB). **MLflow** tracks runs, hyperparameters, metrics (accuracy, loss), and visual artifacts (confusion matrix).
3. **Packaging & API:** The model is served lazily by a **FastAPI** web application to avoid loading overhead during testing. The app is packaged via **Docker**.
4. **CI/CD Automation:** **GitHub Actions** trigger on pushes and PRs. Tests (Pytest) run first; if tests pass, the container image is built and continuously deployed to the Docker registry or designated environment.
5. **Observability:** Performance metrics (like prediction requests, latency) and system health are monitored with **Prometheus** endpoints.

---

## Project Structure
```text
binary-image-classification-mlops/
├── src/                      # Core ML and API logic (train, inference, evaluation)
├── tests/                    # Pytest unit and integration tests
├── models/                   # Tracked by DVC (model.h5)
├── artifacts/                # Outputs, metrics, and plots (MLflow/DVC)
├── configs/                  # Configurations
├── .github/workflows/        # CI/CD pipelines (GitHub Actions)
├── Dockerfile                # Defines the FastAPI inference container
├── docker-compose.yml        # Orchestration containing the app and observability stack
├── requirements.txt          # Pinned dependencies
├── dvc.yaml                  # Data pipeline orchestration definition
└── README.md                 # You're reading this!
```

---

## How to Run the Project Locally

Follow these steps to pull the repository and run the API inference strictly on your local machine.

**1. Clone the repository**
```bash
git clone https://github.com/KSharma-SourceCode/binary-image-classification-mlops.git
cd binary-image-classification-mlops
```

**2. Setup a virtual environment**
```bash
python -m venv .venv
# On Linux/MacOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

**3. Install pinned dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull Large Assets (Models/Data)**
Because the model acts over 100MB, it is stored in an external remote using DVC.
```bash
dvc pull
```

**5. Start the FastAPI Service**
```bash
uvicorn src.inference:app --reload
```
The API is now running at `http://localhost:8000`. You can visit `http://localhost:8000/docs` to interact visually with the Swagger UI.

**6. Run Tests**
Validate that utilities and APIs are working properly safely using Pytest:
```bash
pytest -v
```

---

## Running with Docker
If you want to validate the Docker container locally before relying on the CI/CD deployment or run the application isolated from your system Python environment:

```bash
docker build -t catsdogs-api .
docker run -p 8000:8000 catsdogs-api
```

---

## API Usage Examples

**1. Health Check Endpoint**
```bash
curl -X GET "http://localhost:8000/health"
```

**2. Make a Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/cat_or_dog_image.jpg"
```
*Expected Response:*
```json
{
  "label": "dog",
  "confidence": 0.94,
  "model_version": "v1"
}
```

**3. System Metrics (Prometheus Format)**
```bash
curl -X GET "http://localhost:8000/metrics"
```

---

## CI/CD & Deployment Workflow
We practice continuous integration on every push/PR touching code or dependencies. 

- **Triggers:** Push or PR directly impacting `src/**`, `tests/**`, or `requirements.txt`.
- **Integration Steps:**
  1. Checks out git repository.
  2. Sets up Python 3.10.
  3. Installs requirements.
  4. Runs unit testing suite (`pytest`). Any failing assert strictly blocks the build.
  5. Continerizes the application via the Dockerfile.
- **Delivery Steps:** Deploys directly to the Docker Registry, executing Smoke Tests to validate the health endpoint to avoid service downtimes natively. Look here for [The GitHub Actions Run Flow](https://github.com/KSharma-SourceCode/binary-image-classification-mlops/actions).

---

## Future Improvements
- **Model Registry:** Full MLflow Model Registry integrations mapping aliases to deployments.
- **Automated Retraining Pipeline:** Detect drift naturally, automatically trigger pipeline based on low confidence averages.
- **Deployment Strategy:** Adopt an advanced Canary deployment to slowly route real-world traffic.
- **Autoscaling:** Integrate Kubernetes (HPA) native autoscaling features.
