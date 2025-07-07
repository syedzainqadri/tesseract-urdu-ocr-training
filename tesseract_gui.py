#!/usr/bin/env python3
"""
Tesseract Urdu OCR Training GUI
A complete system for training custom Urdu OCR models using Tesseract 4.x and tesstrain framework
Compatible with macOS
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import subprocess
import threading
import sys
import time
import re
from pathlib import Path
import urllib.request
import shutil

class TesseractTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tesseract Urdu OCR Training System")
        self.root.geometry("800x700")
        
        # Training process variables
        self.training_process = None
        self.is_training = False
        self.training_start_time = None
        self.total_samples = 0
        self.processed_samples = 0
        self.max_iterations = 0
        self.current_iteration = 0
        
        # Setup directories
        self.training_dir = Path.home() / "tesseract_training"
        self.tessdata_dir = self.training_dir / "tessdata"
        self.output_dir = self.training_dir / "output"
        self.tesstrain_dir = self.training_dir / "tesstrain"
        
        self.setup_ui()
        self.check_environment()
    
    def setup_ui(self):
        """Create the main user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Tesseract Urdu OCR Training System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Dataset selection
        ttk.Label(main_frame, text="Ground Truth Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar()
        dataset_frame = ttk.Frame(main_frame)
        dataset_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        dataset_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(dataset_frame, textvariable=self.dataset_var, state='readonly').grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=1)
        
        # Model parameters
        ttk.Label(main_frame, text="Model Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.model_name_var = tk.StringVar(value="urd_custom")
        ttk.Entry(main_frame, textvariable=self.model_name_var).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(main_frame, text="Start Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.start_model_var = tk.StringVar(value="urd")
        ttk.Entry(main_frame, textvariable=self.start_model_var).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(main_frame, text="Max Iterations:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.max_iterations_var = tk.StringVar(value="10000")
        ttk.Entry(main_frame, textvariable=self.max_iterations_var).grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.download_btn = ttk.Button(button_frame, text="Download Sample Dataset", 
                                      command=self.download_sample_dataset)
        self.download_btn.pack(side=tk.LEFT, padx=5)
        
        self.setup_btn = ttk.Button(button_frame, text="Setup Environment", 
                                   command=self.setup_environment)
        self.setup_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(button_frame, text="Start Training", 
                                   command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Training", 
                                  command=self.stop_training, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress and monitoring frame
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress & Monitoring")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)

        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)

        # Progress counters frame
        counters_frame = ttk.Frame(progress_frame)
        counters_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        counters_frame.columnconfigure(1, weight=1)
        counters_frame.columnconfigure(3, weight=1)

        # Sample counter
        ttk.Label(counters_frame, text="Samples Processed:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.samples_processed_var = tk.StringVar(value="0 / 0")
        ttk.Label(counters_frame, textvariable=self.samples_processed_var,
                 font=('Arial', 10, 'bold'), foreground='blue').grid(row=0, column=1, sticky=tk.W)

        # Iteration counter
        ttk.Label(counters_frame, text="Iterations:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.iterations_var = tk.StringVar(value="0 / 0")
        ttk.Label(counters_frame, textvariable=self.iterations_var,
                 font=('Arial', 10, 'bold'), foreground='green').grid(row=0, column=3, sticky=tk.W)

        # Error rate and time
        ttk.Label(counters_frame, text="Error Rate:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.error_rate_var = tk.StringVar(value="N/A")
        ttk.Label(counters_frame, textvariable=self.error_rate_var,
                 font=('Arial', 10, 'bold'), foreground='red').grid(row=1, column=1, sticky=tk.W)

        ttk.Label(counters_frame, text="Training Time:").grid(row=1, column=2, sticky=tk.W, padx=(20, 5))
        self.training_time_var = tk.StringVar(value="00:00:00")
        ttk.Label(counters_frame, textvariable=self.training_time_var,
                 font=('Arial', 10, 'bold'), foreground='purple').grid(row=1, column=3, sticky=tk.W)

        # Status label
        self.status_var = tk.StringVar(value="Ready to start training")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, font=('Arial', 9))
        status_label.grid(row=2, column=0, padx=10, pady=5)
        
        # Log area
        ttk.Label(main_frame, text="Training Logs:").grid(row=7, column=0, sticky=tk.W, pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.log_text.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for resizing
        main_frame.rowconfigure(8, weight=1)

        # Start periodic update timer
        self.update_timer()
    
    def log_message(self, message):
        """Add message to log area"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_progress_counters(self):
        """Update the progress counter displays"""
        # Update samples processed
        if self.total_samples > 0:
            self.samples_processed_var.set(f"{self.processed_samples} / {self.total_samples}")

        # Update iterations
        if self.max_iterations > 0:
            self.iterations_var.set(f"{self.current_iteration} / {self.max_iterations}")

        # Update training time
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.training_time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def reset_progress_counters(self):
        """Reset all progress counters"""
        self.samples_processed_var.set("0 / 0")
        self.iterations_var.set("0 / 0")
        self.error_rate_var.set("N/A")
        self.training_time_var.set("00:00:00")
        self.status_var.set("Ready to start training")
        self.total_samples = 0
        self.processed_samples = 0
        self.max_iterations = 0
        self.current_iteration = 0
        self.training_start_time = None

    def parse_training_output(self, line):
        """Parse training output to extract progress information"""
        try:
            # Parse sample processing (tesseract commands)
            if "tesseract" in line and ".tif" in line and "lstm.train" in line:
                self.processed_samples += 1
                self.status_var.set(f"Processing sample {self.processed_samples}/{self.total_samples}")

            # Parse iteration progress from lstmtraining
            # Example: "At iteration 100/200/200, mean rms=2.5%, delta=10%, BCER train=25%"
            iteration_match = re.search(r'At iteration (\d+)/\d+/\d+.*?BCER train=([0-9.]+)%', line)
            if iteration_match:
                self.current_iteration = int(iteration_match.group(1))
                error_rate = float(iteration_match.group(2))
                self.error_rate_var.set(f"{error_rate:.2f}%")
                self.status_var.set(f"Training iteration {self.current_iteration}/{self.max_iterations}")

            # Parse other progress indicators
            if "Extracting tessdata components" in line:
                self.status_var.set("Extracting base model components...")
            elif "unicharset_extractor" in line:
                self.status_var.set("Extracting character set...")
            elif "combine_lang_model" in line:
                self.status_var.set("Creating language model...")
            elif "lstmtraining" in line and "--traineddata" in line:
                self.status_var.set("Starting LSTM training...")
            elif "Finished!" in line:
                self.status_var.set("Training completed!")
            elif "New best BCER" in line:
                # Extract best error rate
                best_match = re.search(r'New best BCER = ([0-9.]+)', line)
                if best_match:
                    best_error = float(best_match.group(1))
                    self.error_rate_var.set(f"{best_error:.2f}% (best)")

        except Exception as e:
            # Don't let parsing errors stop training
            pass

    def update_timer(self):
        """Periodic timer to update progress counters"""
        if self.is_training and self.training_start_time:
            self.update_progress_counters()

        # Schedule next update in 1 second
        self.root.after(1000, self.update_timer)
    
    def check_environment(self):
        """Check if required tools are installed"""
        self.log_message("Checking environment...")
        
        # Check Tesseract
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.log_message(f"✓ Tesseract found: {result.stdout.split()[1]}")
            else:
                self.log_message("✗ Tesseract not found")
        except FileNotFoundError:
            self.log_message("✗ Tesseract not found - please install first")
        
        # Check directories
        if self.tesstrain_dir.exists():
            self.log_message(f"✓ Tesstrain directory found: {self.tesstrain_dir}")
        else:
            self.log_message(f"✗ Tesstrain directory not found: {self.tesstrain_dir}")
        
        self.log_message("Environment check complete.")
    
    def browse_dataset(self):
        """Browse for ground truth dataset directory"""
        directory = filedialog.askdirectory(
            title="Select Ground Truth Dataset Directory",
            initialdir=str(Path.home())
        )
        if directory:
            self.dataset_var.set(directory)
            self.validate_dataset(directory)
    
    def validate_dataset(self, directory):
        """Validate that the dataset contains .tif and .gt.txt files"""
        path = Path(directory)
        tif_files = list(path.glob("*.tif"))
        gt_files = list(path.glob("*.gt.txt"))
        
        self.log_message(f"Dataset validation:")
        self.log_message(f"  Found {len(tif_files)} .tif files")
        self.log_message(f"  Found {len(gt_files)} .gt.txt files")
        
        if len(tif_files) == 0 or len(gt_files) == 0:
            self.log_message("⚠ Warning: Dataset should contain both .tif and .gt.txt files")
        else:
            self.log_message("✓ Dataset appears valid")

    def download_sample_dataset(self):
        """Download a sample Urdu OCR dataset"""
        self.log_message("Sample dataset download feature coming soon...")
        self.log_message("For now, please prepare your dataset with:")
        self.log_message("  - .tif image files")
        self.log_message("  - corresponding .gt.txt ground truth files")
        self.log_message("  - same base filename for each pair")

    def setup_environment(self):
        """Setup the training environment"""
        def setup_thread():
            try:
                self.progress.start()
                self.setup_btn.config(state='disabled')

                self.log_message("Setting up training environment...")

                # Create directories
                self.training_dir.mkdir(exist_ok=True)
                self.tessdata_dir.mkdir(exist_ok=True)
                self.output_dir.mkdir(exist_ok=True)

                self.log_message(f"✓ Created directory: {self.training_dir}")
                self.log_message(f"✓ Created directory: {self.tessdata_dir}")
                self.log_message(f"✓ Created directory: {self.output_dir}")

                # Clone tesstrain if not exists
                if not self.tesstrain_dir.exists():
                    self.log_message("Cloning tesstrain repository...")
                    result = subprocess.run([
                        'git', 'clone',
                        'https://github.com/tesseract-ocr/tesstrain.git',
                        str(self.tesstrain_dir)
                    ], capture_output=True, text=True, cwd=str(self.training_dir))

                    if result.returncode == 0:
                        self.log_message("✓ Tesstrain repository cloned successfully")
                    else:
                        self.log_message(f"✗ Failed to clone tesstrain: {result.stderr}")
                        return
                else:
                    self.log_message("✓ Tesstrain repository already exists")

                # Download Urdu model if not exists
                urd_model_path = self.tessdata_dir / "urd.traineddata"
                if not urd_model_path.exists():
                    self.log_message("Downloading Urdu base model...")
                    try:
                        urllib.request.urlretrieve(
                            "https://github.com/tesseract-ocr/tessdata_best/raw/main/urd.traineddata",
                            str(urd_model_path)
                        )
                        self.log_message("✓ Urdu base model downloaded successfully")
                    except Exception as e:
                        self.log_message(f"✗ Failed to download Urdu model: {e}")
                        return
                else:
                    self.log_message("✓ Urdu base model already exists")

                self.log_message("✓ Environment setup complete!")

            except Exception as e:
                self.log_message(f"✗ Setup failed: {e}")
            finally:
                self.progress.stop()
                self.setup_btn.config(state='normal')

        threading.Thread(target=setup_thread, daemon=True).start()

    def start_training(self):
        """Start the training process"""
        if self.is_training:
            messagebox.showwarning("Training", "Training is already in progress!")
            return

        # Validate inputs
        if not self.dataset_var.get():
            messagebox.showerror("Error", "Please select a ground truth dataset directory")
            return

        if not Path(self.dataset_var.get()).exists():
            messagebox.showerror("Error", "Selected dataset directory does not exist")
            return

        if not self.tesstrain_dir.exists():
            messagebox.showerror("Error", "Tesstrain not found. Please run 'Setup Environment' first")
            return

        model_name = self.model_name_var.get().strip()
        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return

        # Count total samples in dataset
        dataset_path = Path(self.dataset_var.get())
        tif_files = list(dataset_path.glob("*.tif"))
        self.total_samples = len(tif_files)
        try:
            self.max_iterations = int(self.max_iterations_var.get())
            if self.max_iterations <= 0:
                raise ValueError("Max iterations must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid max iterations value: {e}")
            return

        self.processed_samples = 0
        self.current_iteration = 0

        self.log_message(f"📊 Dataset contains {self.total_samples} samples")
        self.log_message(f"🎯 Training for {self.max_iterations} iterations")

        def training_thread():
            try:
                self.is_training = True
                self.train_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
                self.progress.start()

                # Initialize progress tracking
                self.training_start_time = time.time()
                self.reset_progress_counters()
                self.total_samples = len(tif_files)
                try:
                    self.max_iterations = int(self.max_iterations_var.get())
                except ValueError:
                    self.max_iterations = 10000  # Default fallback
                self.status_var.set("Initializing training...")
                self.update_progress_counters()

                self.log_message("="*50)
                self.log_message("Starting Tesseract training...")
                self.log_message(f"Model name: {model_name}")
                self.log_message(f"Start model: {self.start_model_var.get()}")
                self.log_message(f"Dataset: {self.dataset_var.get()}")
                self.log_message(f"Max iterations: {self.max_iterations_var.get()}")
                self.log_message("="*50)

                # Prepare training command (use gmake for newer version)
                # Note: Removed OUTPUT_DIR to avoid Makefile path conflicts
                cmd = [
                    'gmake', 'training',
                    f'MODEL_NAME={model_name}',
                    f'START_MODEL={self.start_model_var.get()}',
                    'LANG_TYPE=Indic',
                    f'TESSDATA={self.tessdata_dir}',
                    f'GROUND_TRUTH_DIR={self.dataset_var.get()}',
                    f'MAX_ITERATIONS={self.max_iterations_var.get()}'
                ]

                self.log_message(f"Command: {' '.join(cmd)}")

                # Start training process
                self.training_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.tesstrain_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Read output in real-time and parse progress
                for line in iter(self.training_process.stdout.readline, ''):
                    if not self.is_training:  # Check if stopped
                        break

                    line = line.rstrip()
                    self.log_message(line)

                    # Parse progress information
                    self.parse_training_output(line)

                    # Update progress counters periodically
                    self.update_progress_counters()

                # Wait for process to complete
                return_code = self.training_process.wait()

                if return_code == 0 and self.is_training:
                    self.log_message("="*50)
                    self.log_message("✓ Training completed successfully!")

                    # Copy model from default location to output directory
                    import shutil
                    default_model_path = self.tesstrain_dir / "data" / f"{model_name}.traineddata"
                    alt_model_path = self.tesstrain_dir / "data" / self.start_model_var.get() / f"{model_name}.traineddata"

                    model_copied = False
                    if default_model_path.exists():
                        shutil.copy2(default_model_path, self.output_dir)
                        self.log_message(f"✓ Model copied to: {self.output_dir / f'{model_name}.traineddata'}")
                        model_copied = True
                    elif alt_model_path.exists():
                        shutil.copy2(alt_model_path, self.output_dir)
                        self.log_message(f"✓ Model copied to: {self.output_dir / f'{model_name}.traineddata'}")
                        model_copied = True
                    else:
                        self.log_message("⚠️  Model file not found in expected locations")
                        self.log_message(f"   Checked: {default_model_path}")
                        self.log_message(f"   Checked: {alt_model_path}")

                    if model_copied:
                        self.log_message(f"✓ Training completed! Model available in Enhanced OCR app.")
                        messagebox.showinfo("Success", f"Training completed!\nModel saved to: {self.output_dir}\n\nRefresh the Enhanced OCR app to use your new model!")
                    else:
                        self.log_message(f"✓ Training completed! Check tesstrain directory for model.")
                        messagebox.showinfo("Success", f"Training completed!\nCheck: {self.tesstrain_dir}/data/ for model file")
                elif self.is_training:
                    self.log_message("="*50)
                    self.log_message(f"✗ Training failed with return code: {return_code}")
                    messagebox.showerror("Error", "Training failed. Check logs for details.")

            except Exception as e:
                self.log_message(f"✗ Training error: {e}")
                messagebox.showerror("Error", f"Training failed: {e}")
            finally:
                self.is_training = False
                self.train_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
                self.progress.stop()
                self.training_process = None

                # Final progress update
                if self.training_start_time:
                    self.update_progress_counters()
                    if self.current_iteration >= self.max_iterations:
                        self.status_var.set("Training completed!")
                    else:
                        self.status_var.set("Training stopped")

        threading.Thread(target=training_thread, daemon=True).start()

    def stop_training(self):
        """Stop the training process"""
        if self.training_process and self.is_training:
            self.log_message("Stopping training...")
            self.is_training = False
            self.status_var.set("Stopping training...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
            self.log_message("Training stopped by user")
            self.status_var.set("Training stopped by user")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = TesseractTrainingGUI(root)

    # Handle window closing
    def on_closing():
        if app.is_training:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to stop and quit?"):
                app.stop_training()
                root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
