import subprocess
import os
import json
import time
import argparse
import signal
import sys
from pathlib import Path
import datetime
from dotenv import load_dotenv

import sys 
sys.path.append('..')
# Import logger and monitoring functionality
from src.fetch_logs import monitor_training, scrape_log, LogFetcher
from src.exp_logging import BaseLogger, create_logger
import logging
import threading

# Load environment variables
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

class TrainingRunner:
    def __init__(self, output_dir="saves", tracking_backend:str = None, logger=None):
        

        self.process = None
        self.training_id = str(int(time.time()))
        self.output_dir = os.path.join(current_dir, '../LLaMA-Factory', output_dir)
        self._should_stop = False

        self.checkpoints_dir = ""
        self.log_file = ""
        self.is_logger_running = False

        
        # Setup logging
        self.tracking_backend = tracking_backend
        self.logger = logger
        if self.logger is None:
            self.logger = create_logger(tracking_backend)
            self.logger.login()
            self.tracking_backend = self.logger.tracking_backend

    def start_logging(self, training_args=None):

        run_name = f"training_{self.training_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        # Create a config dict from training args
        config = {}
        if training_args:
            config = vars(training_args) if hasattr(training_args, '__dict__') else training_args
        
        print(f"Config: {config}")

        # Start tracking run
        if self.logger.tracking_backend == 'wandb':
            run = self.logger.init_run(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                job_type="training",
                config=config,
                name=run_name
            )
        else:  # mlflow
            run = self.logger.init_run(
                project=os.getenv("MLFLOW_EXPERIMENT_NAME", "training"),
                job_type="training",
                config=config,
                name=run_name
            )

        self.is_logger_running = True
        logging.info(f"Tracking run started")

        
    def start_training(self, command, training_args=None):
        
        # Start logging
        if not self.is_logger_running:
            self.start_logging(training_args)

        # Log training command
        logging.info(f"Starting training with command: {command}")
        
        
        # Create log file for the monitor to read
        self.log_file = os.path.join(self.output_dir, "trainer_log.jsonl")
        
        # Create checkpoints directory
        self.checkpoints_dir = self.output_dir
        # self.checkpoints_dir.mkdir(exist_ok=True)
        
        
        # Start training process
        self.process = subprocess.Popen(
            command,
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            # cwd=os.path.join(current_dir, '../LLaMA-Factory'),
        )
        
        # Log the process start
        self.logger.log_metric("training_started", 1.0)
        
        return self.training_id
    
    def is_running(self):
        """Check if the training process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def kill(self):
        """Kill the training process if it's running."""
        if self.is_running():
            logging.info("Terminating training process...")
            self.process.terminate()
            # Give it some time to terminate gracefully
            time.sleep(2)
            # Force kill if still running
            if self.is_running():
                logging.info("Force killing training process...")
                self.process.kill()
                
            # Log the interruption
            if self.logger:
                self.logger.log_metric("training_interrupted", 1.0)
                
            return True
        return False
    
    def signal_handler(self, sig, frame):
        """Handle signal interrupts like CTRL+C."""
        logging.info("\nReceived interrupt signal. Stopping training...")
        self._should_stop = True
        if self.is_running():
            self.kill()
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def run_training(self, command, monitor_interval=15, upload_timeout=60, max_wait_time=900, trigger_evaluation: bool = True, **kwargs):
        """Run training with the given command and monitor it."""
        # self.setup_signal_handlers()
        
        try:
            training_id = self.start_training(command, kwargs.get('training_args'))
            logging.info(f"Training started with ID: {training_id}")
            logging.info(f"Logs will be tracked with {self.tracking_backend}")
            
            fetcher = LogFetcher(
                log_file_path=self.log_file,
                checkpoint_dir=self.checkpoints_dir,
                logger=self.logger
            )
            
            training_completed = threading.Event()

            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(
                target=monitor_training,
                args=(
                    fetcher, 
                    monitor_interval,
                    180,  # stall_timeout
                    training_completed,
                    upload_timeout
                ),
                daemon=True
            )
            monitor_thread.start()
            
            # Wait for training to complete
            while self.is_running() and not self._should_stop:
                time.sleep(5)
                
                # Check for stdout/stderr output
                if self.process.stdout:
                    for line in iter(self.process.stdout.readline, ''):
                        if not line:
                            break
                        logging.info(f"STDOUT: {line.strip()}")
                
                if self.process.stderr:
                    for line in iter(self.process.stderr.readline, ''):
                        if not line:
                            break
                        logging.error(f"STDERR: {line.strip()}")
            
            # Get exit code
            exit_code = self.process.poll()
            training_completed.set()
            logging.info("Training process completed, waiting for model uploads to finish...")
            
            if not self._should_stop:
                # Training completed naturally
                if exit_code == 0:
                    logging.info("Training completed successfully")
                    if self.logger:
                        self.logger.log_metric("training_completed", 1.0)
                        self.logger.log_metric("training_success", 1.0)
                else:
                    logging.info(f"Training failed with exit code {exit_code}")
                    if self.logger:
                        self.logger.log_metric("training_completed", 1.0)
                        self.logger.log_metric("training_failed", exit_code)
            
            # Wait for monitor thread to clean up
            start_wait_time = time.time()
            monitor_thread.join(timeout=upload_timeout)  # Give extra 30 seconds beyond upload timeout
            
            count = 0

            while monitor_thread.is_alive():
                elapsed_time = time.time() - start_wait_time
                if elapsed_time > max_wait_time:
                    logging.warning(f"Maximum wait time ({max_wait_time/60:.1f} min) exceeded. Uploads may still be in progress.")
                    break

                if count > 0:
                    logging.info(f"Waiting for uploads to complete... {elapsed_time:.1f} seconds elapsed")
            
            if monitor_thread.is_alive():
                logging.warning(f"Monitor thread is still running after {(time.time() - start_wait_time)/60:.1f} minutes. " 
                            f"Uploads may still be in progress, but we'll continue.")
            else:
                logging.info("Model uploads completed successfully")
            
            
            return training_id
            
        except KeyboardInterrupt:
            # This shouldn't be reached thanks to the signal handler,
            # but keeping as a fallback
            logging.info("\nTraining interrupted by user")
            self.kill()
            return self.training_id
            
        except Exception as e:
            logging.info(f"\nError during training: {e}")
            if self.logger:
                self.logger.log_metric("training_error", 1.0)
            self.kill()
            raise
        
        finally:

            # monitor_thread.join(timeout=monitor_interval + 5)
            logging.info("Training process finished. Cleaning up...")
            try:
                scrape_log(fetcher, trigger_evaluation=trigger_evaluation)
            except Exception as e:
                logging.error(f"Error in final scrape_log: {e}")
            # Collect any remaining logs

            # # Always close the run
            if self.logger:
                self.logger.finish_run()


def main():
    parser = argparse.ArgumentParser(description="Training CLI wrapper")
    parser.add_argument("--command", default='python ../test/fake_train.py',  help="Training command to run")
    parser.add_argument("--output_dir", default="../temp", help="Output directory")
    parser.add_argument("--monitor_interval", type=int, default=15, 
                        help="Interval in seconds between monitoring checks")
    parser.add_argument("--tracking_backend", choices=["wandb", "mlflow"], 
                        default=os.getenv("TRACKING_BACKEND", "wandb"),
                        help="Tracking backend to use")
    args = parser.parse_args()
    
    # Initialize logger
    logger = create_logger(args.tracking_backend)
    logger.login()
    
    # Create and run the training runner
    runner = TrainingRunner(
        output_dir=args.output_dir,
        tracking_backend=args.tracking_backend,
        logger=logger
    )
    
    runner.run_training(
        args.command, 
        monitor_interval=args.monitor_interval,
        training_args=args
    )


if __name__ == "__main__":
    main()