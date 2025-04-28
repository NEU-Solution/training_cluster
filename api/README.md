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

## Get API Information

```bash
curl -X GET http://localhost:23478/
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
curl -X POST http://localhost:23478/train \
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
curl -X POST http://localhost:23478/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_name": "initial-sft",
    "lora_version": "2",
    "dataset_version": "v1.0",
    "template": "qwen",
    "cutoff_len": 2048,
    "max_samples": 10000,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "save_steps": 501,
    "learning_rate": "2.0e-5",
    "num_epochs": 2.0,
    "tracking_backend": "mlflow",
    "save_name": "test_reasoning"
  }'
```

### Start a training job with reject strategy

```bash
curl -X POST "http://localhost:23478/train?strategy=reject" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_version": "v1.0"
  }'
```

### Get status of a specific job

```bash
curl -X GET https://yt036afigd5k.share.zrok.io/train/43c669ec-3a40-4187-9143-5221614a0d09
```

### Get all training jobs

```bash
curl -X GET http://localhost:23478/train
```

### Cancel a queued job

```bash
curl -X DELETE http://localhost:23478/train/550e8400-e29b-41d4-a716-446655440000
```

## Queue Management

### Get queue status

```bash
curl -X GET http://localhost:23478/queue
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