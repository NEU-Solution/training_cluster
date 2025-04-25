# Training API Curl Examples

Here are curl commands for interacting with all available endpoints in the Training API:
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

### Start a new training job (basic)

```bash
curl -X POST http://localhost:23478/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_name": "initial-sft",
    "dataset_version": "v1.0",  
    "template": "qwen",
    "logging_backend": "mlflow"
  }'
```

### Start a training job with all parameters

```bash
curl -X POST http://localhost:23478/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_version": "v1.0",
    "template": "qwen",
    "cutoff_len": 2048,
    "max_samples": 1000,
    "batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2.0e-5,
    "num_epochs": 3.0,
    "logging_backend": "wandb",
    "webhook_url": "https://example.com/webhook"
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
curl -X GET http://localhost:23478/train/9212c3b8-78dd-4f04-8361-bc553fc02c71
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