import subprocess
import os
import json
import time
import argparse
import signal
import sys
from pathlib import Path


class TrainingRunner:
    def __init__(self, output_dir="./outputs"):
        self.process = None
        self.training_id = str(int(time.time()))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._should_stop = False
        
    def start_training(self, command):
        """Start the training process with the given command."""
        # Start training process
        self.process = subprocess.Popen(
            command,
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        return self.training_id
    
    def is_running(self):
        """Check if the training process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def kill(self):
        """Kill the training process if it's running."""
        if self.is_running():
            print("Terminating training process...")
            self.process.terminate()
            # Give it some time to terminate gracefully
            time.sleep(2)
            # Force kill if still running
            if self.is_running():
                print("Force killing training process...")
                self.process.kill()
            return True
        return False
    
    def signal_handler(self, sig, frame):
        """Handle signal interrupts like CTRL+C."""
        print("\nReceived interrupt signal. Stopping training...")
        self._should_stop = True
        if self.is_running():
            self.kill()
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def run_training(self, command, monitor_interval=15, moinitor_function=None, **kwargs):
        """Run training with the given command and monitor it."""
        self.setup_signal_handlers()
        
        try:
            training_id = self.start_training(command)
            print(f"Training started with ID: {training_id}")
            print("Press Ctrl+C to stop training")
            
            # Monitoring loop
            while self.is_running() and not self._should_stop:
                if moinitor_function is not None:
                    # Call the monitoring function if provided
                    moinitor_function(**kwargs)
                
                time.sleep(monitor_interval)
            
            if not self._should_stop:
                print("Training completed successfully")
            return training_id
            
        except KeyboardInterrupt:
            # This shouldn't be reached thanks to the signal handler,
            # but keeping as a fallback
            print("\nTraining interrupted by user")
            self.kill()
            return self.training_id
            
        except Exception as e:
            print(f"\nError during training: {e}")
            self.kill()
            raise


def main():
    parser = argparse.ArgumentParser(description="Training CLI wrapper")
    parser.add_argument("--command", required=True, help="Training command to run")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--monitor_interval", type=int, default=15, 
                        help="Interval in seconds between monitoring checks")
    args = parser.parse_args()
    
    runner = TrainingRunner(output_dir=args.output_dir)
    runner.run_training(args.command, monitor_interval=args.monitor_interval)


if __name__ == "__main__":
    main()