# Multimodal Clinical AI Platform

An end-to-end clinical AI system for cardiomegaly prediction from chest X-rays and pre-imaging clinical indication text.

The project covers the full machine-learning lifecycle: leakage-aware dataset construction, controlled model comparison, multimodal fusion, failure analysis, calibration, reusable inference, API serving, automated testing, containerization, CI/CD, AWS deployment, and production-style monitoring.

## Key Results

Final evaluation was performed once on a locked, untouched report-level test set.

| Model | AUROC | PR-AUC |
|---|---:|---:|
| Image-only ResNet-18 | 0.8584 | 0.4026 |
| Text-only TF-IDF + Logistic Regression | 0.6676 | 0.1871 |
| **Multimodal Fusion** | **0.8755** | **0.4110** |

The final multimodal model uses fixed late fusion:

```text
0.80 × image probability + 0.20 × text probability
```

Compared with the image-only model, fusion improved test AUROC by **+0.0171** and PR-AUC by **+0.0084**.

See [`docs/results_summary.md`](docs/results_summary.md) for calibration and failure-mode analysis.

## Clinical Framing

The prediction target is **cardiomegaly**.

Inputs:

- frontal chest X-ray
- clinical indication text available before imaging

The indication field was deliberately used instead of radiology findings or impression text to reduce label leakage from post-imaging information.

Dataset splitting is performed at the **report level**, preventing images from the same radiology report from appearing across train, validation, and test sets.

## Model Architecture

### Image branch

- Pretrained ResNet-18
- ImageNet-normalized 224 × 224 chest X-rays
- `layer4` and classification head fine-tuned
- Binary cardiomegaly probability output

### Text branch

- TF-IDF features
- unigrams + bigrams
- logistic regression classifier
- clinical indication text only

Transformer-based clinical text models were also evaluated, but the TF-IDF + logistic regression baseline performed better on validation data and was selected for the final system.

### Multimodal fusion

Image and text probabilities are combined using fixed late fusion:

```text
fusion = 0.80 × image_probability + 0.20 × text_probability
```

The fusion weight was selected using validation data before final test evaluation.

## System Architecture

```text
Chest X-ray ───────→ ResNet-18 ───────→ Image probability ─┐
                                                          │
Clinical indication → TF-IDF + LR ────→ Text probability ─┤
                                                          ↓
                                                  80/20 late fusion
                                                          ↓
                                               Cardiomegaly probability
```

The trained models are exposed through a FastAPI inference service and deployed as a Docker container on AWS ECS/Fargate.

See:

- [`docs/architecture.mmd`](docs/architecture.mmd)
- [`docs/deployment_architecture.mmd`](docs/deployment_architecture.mmd)

## API

The FastAPI service exposes:

```text
GET /health
POST /predict
```

`POST /predict` accepts:

- a PNG or JPEG chest X-ray
- a clinical indication string

Example response:

```json
{
  "image_probability": 0.0221,
  "text_probability": 0.1216,
  "fusion_probability": 0.0420,
  "fusion_alpha": 0.8
}
```

Input validation includes:

- empty indication rejection
- image media-type validation
- image-content verification
- temporary-file cleanup
- generic server-side prediction error handling

Model artifacts are loaded lazily so health checks do not require the full ML stack to initialize.

## Testing

The project includes automated tests for:

- report-level split integrity
- health endpoint behavior
- valid prediction requests
- empty clinical indications
- unsupported media types
- corrupted image uploads

API tests use dependency injection with a lightweight fake predictor, allowing CI to validate API behavior without loading PyTorch or production model artifacts.

## Containerization

The inference service is packaged with Docker.

The production image contains:

- FastAPI application
- inference code
- PyTorch / torchvision runtime
- trained ResNet checkpoint
- TF-IDF vectorizer
- logistic regression classifier

The deployed container targets **Linux ARM64** for the AWS Fargate runtime.

## CI/CD

GitHub Actions implements automated continuous integration and deployment.

On pushes to `main`:

```text
Git push
   ↓
Automated tests
   ↓
GitHub OIDC authentication to AWS
   ↓
Download model artifacts from private S3
   ↓
Build Linux/ARM64 Docker image
   ↓
Push image to Amazon ECR
   ↓
Force new ECS deployment
   ↓
Wait for ECS service stability
```

AWS authentication uses GitHub OIDC rather than long-lived AWS credentials.

Docker images are tagged with both:

- `latest`
- the Git commit SHA

This provides a deployable current image while retaining immutable image versions for traceability.

## AWS Deployment

Production infrastructure is hosted in AWS Frankfurt (`eu-central-1`).

Services used:

- **Amazon S3** for private model-artifact storage
- **Amazon ECR** for Docker images
- **Amazon ECS / Fargate** for container execution
- **Amazon CloudWatch** for centralized logs and resource monitoring

The FastAPI service runs on port `8000`.

The current portfolio deployment uses a public Fargate task IP. Because the task IP can change when ECS replaces a task, it is intentionally not hard-coded in this README.

## Monitoring

CloudWatch captures:

- container startup/runtime logs
- HTTP requests
- request paths
- HTTP status codes
- request latency

Application middleware emits request logs such as:

```text
request method=GET path=/health status=200 duration_ms=1.27
```

Clinical indication text and image contents are intentionally excluded from request logs.

CloudWatch alarms monitor high:

- CPU utilization
- memory utilization

## Repository Structure

```text
clinical-ai_platform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
├── docs/
│   ├── architecture.mmd
│   ├── deployment_architecture.mmd
│   └── results_summary.md
├── scripts/
├── src/
│   ├── api.py
│   └── srcinference.py
├── tests/
├── Dockerfile
├── requirements-api.txt
└── README.md
```

Large datasets, trained checkpoints, prediction outputs, and serialized model artifacts are excluded from Git.

## Design Decisions

Several choices were intentionally kept simple:

- report-level splitting instead of image-level splitting to reduce leakage
- pre-imaging indication text instead of findings/impression text
- late probability fusion instead of a larger joint multimodal network
- dependency-injected API tests instead of loading production models in CI
- Docker + ECS/Fargate instead of Kubernetes
- private S3 model storage instead of committing trained artifacts to Git

The goal was to build a reproducible, testable, deployable clinical AI system without adding infrastructure that was not justified by the project.

## Disclaimer

This project is a research and engineering prototype. It is **not a medical device and is not intended for clinical use or medical decision-making**.