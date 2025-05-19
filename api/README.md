# Training API Curl Examples

Collecting workspace information# Training API Documentation

The Training API provides a RESTful interface for managing LLM fine-tuning jobs. It supports job queuing, tracking, status monitoring, and webhooks for notifications.

## Overview

This API allows you to:
- Submit fine-tuning jobs for language models
- Queue jobs when resources are occupied
- Monitor training progress
- Cancel queued jobs
- Track all historical training runs

## Start API service locally
```bash
cd training_cluster/api
uvicorn train_server:app --host 0.0.0.0 --port 23478 --reload
```
Now the API service should be on http://localhost:23478
For the completion of the project, it is published as https://train_api.quanghung20gg.site

## Get API Information

```bash
curl -X GET https://train_api.quanghung20gg.site
```

## Training Management Commands

### Request Parameters

Key training parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Base model to use | Qwen/Qwen2.5-1.5B-Instruct |
| `dataset_version` | Dataset version | v1.0 |
| `template` | Prompt template format | qwen |
| `learning_rate` | Learning rate | 2.0e-5 |
| `num_epochs` | Training epochs | 3.0 |
| `batch_size` | Batch size | 1 |
| `tracking_backend` | Tracking system (mlflow/wandb) | wandb |
| `webhook_url` | Notification URL | None |

### Start a new training job (basic)

```bash
curl -X POST https://train_api.quanghung20gg.site/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_name": "initial-sft",
    "lora_version": "2",
    "dataset_version": "v1.0",  
    "template": "qwen",
    "tracking_backend": "mlflow"
  }'
```

### Start a training job with all parameters

```bash
curl -X POST https://train_api.quanghung20gg.site/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen3-1.7B",
    "lora_name": "pretrain",
    "lora_version": "1",
    "dataset_version": "v2.0",
    "template": "qwen",
    "cutoff_len": 4096,
    "max_samples": 50000,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "save_steps": 2001,
    "learning_rate": "2.0e-5",
    "num_epochs": 2.0,
    "tracking_backend": "mlflow",
    "save_name": "sft-v4"
  }'
```

### Start a training job with reject strategy

```bash
curl -X POST "https://train_api.quanghung20gg.site/train?strategy=reject" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_version": "v1.0"
  }'
```

### Get status of a specific job

```bash
curl -X GET https://train_api.quanghung20gg.site/train/[JOB_ID]
```

### Get all training jobs

```bash
curl -X GET https://train_api.quanghung20gg.site/train
```

### Cancel a queued job

```bash
curl -X DELETE curl -X GET https://train_api.quanghung20gg.site/train/550e8400-e29b-41d4-a716-446655440000
```

## Queue Management

### Get queue status

```bash
curl -X GET curl -X GET https://train_api.quanghung20gg.site/queue
```

## Response Examples

### Training job creation response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "message": "Training job started"
}
```

### Job status response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "config": {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_version": "v1.0",
    "template": "qwen",
    "cutoff_len": 2048,
    "max_samples": 1000,
    "batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 0.00002,
    "num_epochs": 3.0,
    "tracking_backend": "wandb"
  },
  "start_time": 1712334054.539546
}
```