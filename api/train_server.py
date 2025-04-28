import os
import sys
import uuid
import time
import yaml
import logging
import asyncio
import datetime
import concurrent.futures
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import uvicorn
from copy import deepcopy

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field

# Add parent directory to path for imports
# Add parent directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.train import train
from src.exp_logging import create_logger
from const import *
from utils import load_config_from_yaml

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

app = FastAPI(title="Model Training API", description="API for running LLM training jobs")

# Global thread pool for running CPU-intensive tasks
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# Global state for tracking jobs
training_jobs = {}
is_training_running = False
training_queue = []

# Configuration
CONCURRENCY_STRATEGY = os.getenv("CONCURRENCY_STRATEGY", ConcurrencyStrategy.QUEUE)

@app.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks,
    strategy: ConcurrencyStrategy = Query(
        default=ConcurrencyStrategy.QUEUE,
        description="Strategy for handling concurrent training requests"
    )
):
    """
    Start a training job with the specified parameters.
    """
    global is_training_running

    job_id = str(uuid.uuid4())
    
    # If a config file is specified, load it
    config = request.dict(exclude={"config_path", "webhook_url"})
    
    # Check if there's already a training job running
    if is_training_running:
        if strategy == ConcurrencyStrategy.REJECT:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A training job is already running. Please try again later."
            )
        elif strategy == ConcurrencyStrategy.QUEUE:
            # Add to queue
            training_queue.append((job_id, request, config))
            
            # Create job entry in tracking dict
            training_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "config": config,
                "start_time": time.time(),
                "queue_position": len(training_queue)
            }
            
            return TrainingResponse(
                job_id=job_id,
                status="queued",
                message=f"Training job added to queue at position {len(training_queue)}"
            )
    
    # Start the training
    is_training_running = True
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "running", 
        "config": config,
        "start_time": time.time()
    }
    
    # Start the training in the background without blocking
    background_tasks.add_task(
        run_training_job, 
        job_id=job_id,
        config=config,
        webhook_url=request.webhook_url
    )
    
    # Return immediately with the job ID
    return TrainingResponse(
        job_id=job_id,
        status="started",
        message="Training job started"
    )

async def run_training_job(job_id: str, config: Dict[str, Any], webhook_url: Optional[HttpUrl] = None):
    """
    Run the training job and update the status.
    """
    global is_training_running, training_queue
    
    tracking_backend = config["tracking_backend"]
    logger_instance = None
    tracking_url = None
    
    try:
        logging.info(f"Starting training job {job_id} with tracking backend: {tracking_backend}")
        
        # Initialize the tracking logger
        logger_instance = create_logger(tracking_backend)
        logger_instance.login()
        
        # Initialize tracking run
        run_name = f"api_train_{datetime.datetime.now().strftime('%Y-%m-%d')}_{job_id[:8]}"
        
        if tracking_backend == 'wandb':
            import wandb
            wandb.login(key=WANDB_API_KEY)
            run = logger_instance.init_run(
                project=config.get("wandb_project"),
                entity=config.get("wandb_entity"),
                job_type="api_training",
                config=config,
                name=run_name
            )
            if hasattr(wandb, "run") and wandb.run is not None:
                tracking_url = wandb.run.get_url()
        else:  # mlflow
            run = logger_instance.init_run(
                project=config.get("mlflow_experiment_name", "training"),
                job_type="api_training",
                config=config,
                name=run_name
            )
            # Get MLflow tracking URL if available
            if hasattr(logger_instance, 'get_tracking_url'):
                tracking_url = logger_instance.get_tracking_url()

        
        # Update job with tracking URL
        training_jobs[job_id]["tracking_url"] = tracking_url

        config["learning_rate"] = str(config["learning_rate"])  # Convert to string as expected by train function
        
        # pass the logger instance to the training function
        train_config = deepcopy(config)
        train_config['logger'] = logger_instance
        # Run the training in a separate thread to not block the event loop
        loop = asyncio.get_running_loop()
        output_path = await loop.run_in_executor(
            thread_pool,
            lambda: train(
                **train_config,
            )
        )
        
        # Update job status
        training_jobs[job_id].update({
            "status": "completed",
            "output_path": output_path,
            "end_time": time.time()
        })
        
        logging.info(f"Training job {job_id} completed successfully")
        
        # Send webhook notification if URL was provided
        if webhook_url:
            await send_webhook_notification(
                webhook_url=webhook_url,
                job_id=job_id,
                status="completed",
                tracking_url=tracking_url,
                output_path=output_path
            )
    
    except Exception as e:
        logging.exception(f"Error in training job {job_id}: {str(e)}")
        # Update job status with error
        training_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": time.time()
        })
        
        # Send webhook notification about failure
        if webhook_url:
            await send_webhook_notification(
                webhook_url=webhook_url,
                job_id=job_id,
                status="failed",
                error=str(e)
            )
    
    finally:
        # Finish the tracking run if it was started
        if logger_instance:
            logger_instance.finish_run()
        
        # Set global flag to allow new training jobs
        is_training_running = False
        
        # Process the next job in the queue if any
        await process_queue()

async def process_queue():
    """
    Process the next job in the queue if available.
    """
    global is_training_running, training_queue
    
    if training_queue and not is_training_running:
        job_id, request, config = training_queue.pop(0)
        
        # Update remaining queue positions
        for i, (queued_job_id, _, _) in enumerate(training_queue):
            if queued_job_id in training_jobs:
                training_jobs[queued_job_id]["queue_position"] = i + 1
        
        # Start the job
        is_training_running = True
        training_jobs[job_id].update({
            "status": "running",
            "start_time": time.time()  # Update start time to actual execution time
        })
        
        # Run the job
        await run_training_job(
            job_id=job_id,
            config=config,
            webhook_url=request.webhook_url if hasattr(request, "webhook_url") else None
        )

async def send_webhook_notification(webhook_url: HttpUrl, **data):
    """
    Send a webhook notification with job status and results.
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(str(webhook_url), json=data) as response:
                if response.status >= 400:
                    logging.error(f"Failed to send webhook notification: {response.status} {await response.text()}")
    except Exception as e:
        logging.error(f"Error sending webhook notification: {str(e)}")

@app.get("/train/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """
    Get the status of a specific training job.
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )
    
    return TrainingStatus(**training_jobs[job_id])

@app.get("/train", response_model=List[TrainingStatus])
async def get_all_trainings():
    """
    Get a list of all training jobs.
    """
    return [TrainingStatus(**job_data) for job_data in training_jobs.values()]

@app.get("/queue")
async def get_queue_status():
    """
    Get the current queue status.
    """
    return {
        "is_training_running": is_training_running,
        "queue_length": len(training_queue),
        "queued_jobs": [job_id for job_id, _, _ in training_queue]
    }

@app.delete("/train/{job_id}")
async def cancel_training(job_id: str):
    """
    Cancel a queued training job. Currently running jobs cannot be cancelled.
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )
    
    job_status = training_jobs[job_id]["status"]
    
    if job_status == "queued":
        # Remove from queue
        for i, (queued_job_id, _, _) in enumerate(training_queue):
            if queued_job_id == job_id:
                training_queue.pop(i)
                training_jobs[job_id].update({
                    "status": "cancelled",
                    "end_time": time.time()
                })
                
                # Update remaining queue positions
                for j, (remaining_job_id, _, _) in enumerate(training_queue):
                    if remaining_job_id in training_jobs:
                        training_jobs[remaining_job_id]["queue_position"] = j + 1
                
                return {"status": "cancelled", "message": f"Training job {job_id} has been cancelled"}
    
    elif job_status == "running":
        return JSONResponse(
            status_code=400,
            content={"detail": f"Cannot cancel running job {job_id}. Feature not implemented."}
        )
    
    else:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Job {job_id} has status {job_status} and cannot be cancelled"}
        )

@app.get("/")
async def root():
    """
    Root endpoint, showing API information.
    """
    return {
        "name": "Model Training API",
        "version": "1.0.0",
        "description": "API for running LLM fine-tuning jobs",
        "endpoints": {
            "POST /train": "Start a new training job",
            "GET /train/{job_id}": "Get status of a specific job",
            "GET /train": "List all jobs",
            "GET /queue": "Get queue status",
            "DELETE /train/{job_id}": "Cancel a queued job"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=23478)