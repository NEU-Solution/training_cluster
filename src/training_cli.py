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
from src.fetch_logs import monitor_training
from src.exp_logging import BaseLogger, create_logger
import logging

# Load environment variables
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

class TrainingRunner:
    def __init__(self, output_dir="saves", tracking_backend="wandb", logger=None):
        self.process = None
        self.training_id = str(int(time.time()))
        self.output_dir = os.path.join(current_dir, '../LLaMA-Factory', output_dir)
        self._should_stop = False

        self.checkpoints_dir = ""
        self.log_file = ""

        
        # Setup logging
        self.tracking_backend = tracking_backend
        self.logger = logger
        if self.logger is None:
            self.logger = create_logger(tracking_backend)
            self.logger.login()
        
    def start_training(self, command, training_args=None):
        """Start the training process with the given command."""
        # Initialize tracking run
        run_name = f"training_{self.training_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        # Create a config dict from training args
        config = {}
        if training_args:
            config = vars(training_args) if hasattr(training_args, '__dict__') else training_args
        
        # Add command to config
        config['command'] = command
        
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
        logging.info(f"Tracking run started")
        
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
        
    def run_training(self, command, monitor_interval=15, **kwargs):
        """Run training with the given command and monitor it."""
        self.setup_signal_handlers()
        
        try:
            training_id = self.start_training(command, kwargs.get('training_args'))
            logging.info(f"Training started with ID: {training_id}")
            logging.info(f"Logs will be tracked with {self.tracking_backend}")
            
            # Start monitoring in a separate thread
            from threading import Thread
            monitor_thread = Thread(
                target=monitor_training,
                args=(self.log_file, self.checkpoints_dir, self.logger, monitor_interval),
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
            monitor_thread.join(timeout=5)
            
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
                self.logger.update_summary("error_message", str(e))
            self.kill()
            raise
        
        finally:
            # Always close the run
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