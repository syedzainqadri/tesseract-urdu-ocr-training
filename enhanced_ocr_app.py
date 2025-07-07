#!/usr/bin/env python3
"""
Enhanced Production Urdu OCR Application
Advanced image processing and OCR with support for multiple formats and quality enhancement
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
from pathlib import Path
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
import threading
import cv2
import numpy as np

class EnhancedOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Urdu OCR Application with Image Processing")
        self.root.geometry("1400x900")
        
        # Model paths
        self.models_dir = Path.home() / "tesseract_training" / "output"
        self.tessdata_dir = Path.home() / "tesseract_training" / "tessdata"
        self.current_model = None
        self.current_image = None
        self.processed_image = None
        self.original_image = None
        
        self.setup_ui()
        self.scan_for_models()
    
    def setup_ui(self):
        """Create the enhanced user interface"""
        # Main frame with notebook for tabs
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Enhanced Urdu OCR with Image Processing", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # OCR Tab
        self.create_ocr_tab(notebook)
        
        # Image Processing Tab
        self.create_processing_tab(notebook)
        
        # Batch Processing Tab
        self.create_batch_tab(notebook)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select a model and image to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_ocr_tab(self, notebook):
        """Create the main OCR tab"""
        ocr_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ocr_frame, text="OCR Processing")
        
        ocr_frame.columnconfigure(1, weight=1)
        ocr_frame.rowconfigure(3, weight=1)
        
        # Model selection
        ttk.Label(ocr_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar()
        model_frame = ttk.Frame(ocr_frame)
        model_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(0, weight=1)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state='readonly')
        self.model_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        ttk.Button(model_frame, text="Refresh", command=self.scan_for_models).grid(row=0, column=1)
        
        # Image selection
        ttk.Label(ocr_frame, text="Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        image_frame = ttk.Frame(ocr_frame)
        image_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        image_frame.columnconfigure(0, weight=1)
        
        self.image_var = tk.StringVar()
        ttk.Entry(image_frame, textvariable=self.image_var, state='readonly').grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(image_frame, text="Browse", command=self.browse_image).grid(row=0, column=1)
        
        # Processing options
        options_frame = ttk.LabelFrame(ocr_frame, text="Image Enhancement Options")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.auto_enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Auto-enhance image quality", 
                       variable=self.auto_enhance_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Remove noise", 
                       variable=self.denoise_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.deskew_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Auto-deskew", 
                       variable=self.deskew_var).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        
        self.upscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Upscale low-res images", 
                       variable=self.upscale_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(ocr_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        self.process_btn = ttk.Button(button_frame, text="Process Image", 
                                     command=self.process_image, state='disabled')
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.ocr_btn = ttk.Button(button_frame, text="Extract Text (OCR)", 
                                 command=self.perform_ocr, state='disabled')
        self.ocr_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Save Text", command=self.save_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Results area
        results_frame = ttk.Frame(ocr_frame)
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Image preview with before/after
        image_notebook = ttk.Notebook(results_frame)
        image_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Original image tab
        self.original_frame = ttk.Frame(image_notebook)
        image_notebook.add(self.original_frame, text="Original")
        self.original_frame.columnconfigure(0, weight=1)
        self.original_frame.rowconfigure(0, weight=1)
        
        self.original_label = ttk.Label(self.original_frame, text="No image selected")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Processed image tab
        self.processed_frame = ttk.Frame(image_notebook)
        image_notebook.add(self.processed_frame, text="Processed")
        self.processed_frame.columnconfigure(0, weight=1)
        self.processed_frame.rowconfigure(0, weight=1)
        
        self.processed_label = ttk.Label(self.processed_frame, text="No processed image")
        self.processed_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Text results
        text_frame = ttk.LabelFrame(results_frame, text="Extracted Text")
        text_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.text_result = scrolledtext.ScrolledText(text_frame, height=20, width=50, wrap=tk.WORD)
        self.text_result.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
    
    def create_processing_tab(self, notebook):
        """Create the image processing tab"""
        proc_frame = ttk.Frame(notebook, padding="10")
        notebook.add(proc_frame, text="Image Processing")
        
        ttk.Label(proc_frame, text="Advanced image processing controls will be here", 
                 font=('Arial', 12)).pack(pady=20)
        
        # Manual processing controls
        controls_frame = ttk.LabelFrame(proc_frame, text="Manual Controls")
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Brightness
        ttk.Label(controls_frame, text="Brightness:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(controls_frame, from_=0.5, to=2.0, variable=self.brightness_var, orient=tk.HORIZONTAL)
        brightness_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Contrast
        ttk.Label(controls_frame, text="Contrast:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(controls_frame, from_=0.5, to=2.0, variable=self.contrast_var, orient=tk.HORIZONTAL)
        contrast_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        controls_frame.columnconfigure(1, weight=1)
    
    def create_batch_tab(self, notebook):
        """Create the batch processing tab"""
        batch_frame = ttk.Frame(notebook, padding="10")
        notebook.add(batch_frame, text="Batch Processing")
        
        ttk.Label(batch_frame, text="Batch process multiple images", 
                 font=('Arial', 12)).pack(pady=20)
        
        # Batch controls
        batch_controls = ttk.Frame(batch_frame)
        batch_controls.pack(fill=tk.X, pady=10)
        
        ttk.Button(batch_controls, text="Select Folder", command=self.select_batch_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(batch_controls, text="Process All", command=self.process_batch).pack(side=tk.LEFT, padx=5)
        
        # Batch results
        self.batch_results = scrolledtext.ScrolledText(batch_frame, height=15)
        self.batch_results.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def scan_for_models(self):
        """Scan for available trained models"""
        models = []
        
        # Check for custom trained models
        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*.traineddata"):
                models.append(f"custom:{model_file.stem}")
        
        # Check for models in tessdata directory
        if self.tessdata_dir.exists():
            for model_file in self.tessdata_dir.glob("*.traineddata"):
                models.append(f"tessdata:{model_file.stem}")
        
        # Add system models
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                system_langs = result.stdout.strip().split('\n')[1:]
                for lang in system_langs:
                    if lang.strip():
                        models.append(f"system:{lang.strip()}")
        except:
            pass
        
        self.model_combo['values'] = models
        if models:
            # Prefer custom models, then tessdata, then system
            for model in models:
                if 'urd_custom' in model:
                    self.model_combo.set(model)
                    break
                elif 'urd' in model and 'tessdata:' in model:
                    self.model_combo.set(model)
                elif 'urd' in model and not self.model_combo.get():
                    self.model_combo.set(model)
            
            if not self.model_combo.get() and models:
                self.model_combo.set(models[0])
            
            self.status_var.set(f"Found {len(models)} models")
        else:
            self.status_var.set("No models found")
        
        self.check_ready_state()
    
    def on_model_change(self, event=None):
        """Handle model selection change"""
        model = self.model_var.get()
        if 'urd_custom' in model:
            self.status_var.set("Using your custom trained model!")
        elif 'urd' in model:
            self.status_var.set("Using Urdu base model")
        else:
            self.status_var.set(f"Using model: {model}")
        self.check_ready_state()
    
    def browse_image(self):
        """Browse for image file with support for multiple formats"""
        filetypes = [
            ("All supported", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.gif *.webp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("TIFF files", "*.tiff *.tif"),
            ("WebP files", "*.webp"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image for OCR",
            filetypes=filetypes
        )
        
        if filename:
            self.image_var.set(filename)
            self.current_image = filename
            self.load_original_image()
            self.check_ready_state()
    
    def load_original_image(self):
        """Load and display original image"""
        if not self.current_image:
            return
        
        try:
            # Load image with PIL (supports more formats)
            self.original_image = Image.open(self.current_image)
            
            # Convert to RGB if necessary
            if self.original_image.mode in ('RGBA', 'LA', 'P'):
                self.original_image = self.original_image.convert('RGB')
            
            # Create preview
            preview_image = self.original_image.copy()
            preview_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(preview_image)
            self.original_label.configure(image=photo, text="")
            self.original_label.image = photo
            
            # Get image info
            width, height = self.original_image.size
            mode = self.original_image.mode
            format_name = self.original_image.format or "Unknown"
            
            self.status_var.set(f"Loaded: {Path(self.current_image).name} ({width}x{height}, {mode}, {format_name})")
            
        except Exception as e:
            self.original_label.configure(image="", text=f"Error loading image: {e}")
            self.status_var.set("Error loading image")
    
    def check_ready_state(self):
        """Check if ready to perform operations"""
        has_model = bool(self.model_var.get())
        has_image = bool(self.current_image)

        self.process_btn.configure(state='normal' if has_image else 'disabled')
        self.ocr_btn.configure(state='normal' if (has_model and has_image) else 'disabled')

    def process_image(self):
        """Process image with enhancement options"""
        if not self.original_image:
            return

        try:
            self.status_var.set("Processing image...")

            # Start with original image
            processed = self.original_image.copy()

            # Convert to numpy array for OpenCV processing
            cv_image = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)

            # Auto-enhance if enabled
            if self.auto_enhance_var.get():
                processed = self.auto_enhance_image(processed)
                cv_image = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)

            # Upscale if enabled and image is small
            if self.upscale_var.get():
                height, width = cv_image.shape[:2]
                if width < 1000 or height < 1000:
                    scale_factor = max(1000/width, 1000/height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Denoise if enabled
            if self.denoise_var.get():
                cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)

            # Deskew if enabled
            if self.deskew_var.get():
                cv_image = self.deskew_image(cv_image)

            # Convert back to PIL
            self.processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Display processed image
            preview_processed = self.processed_image.copy()
            preview_processed.thumbnail((400, 400), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(preview_processed)
            self.processed_label.configure(image=photo, text="")
            self.processed_label.image = photo

            self.status_var.set("Image processing complete")

        except Exception as e:
            self.status_var.set(f"Processing error: {e}")
            messagebox.showerror("Processing Error", f"Failed to process image: {e}")

    def auto_enhance_image(self, image):
        """Automatically enhance image for better OCR"""
        # Convert to grayscale for analysis
        gray = image.convert('L')

        # Auto-contrast
        enhanced = ImageOps.autocontrast(gray, cutoff=2)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.2)

        # Convert back to RGB
        if image.mode == 'RGB':
            enhanced = enhanced.convert('RGB')

        return enhanced

    def deskew_image(self, cv_image):
        """Automatically deskew image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert if necessary (text should be black on white)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)

            # Find contours and get the largest ones
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour (likely text area)
                largest_contour = max(contours, key=cv2.contourArea)

                # Get minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]

                # Correct angle
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90

                # Only rotate if angle is significant
                if abs(angle) > 0.5:
                    # Get rotation matrix
                    (h, w) = cv_image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)

                    # Perform rotation
                    rotated = cv2.warpAffine(cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated

            return cv_image

        except Exception as e:
            print(f"Deskew error: {e}")
            return cv_image

    def perform_ocr(self):
        """Perform OCR on the image"""
        if not self.model_var.get():
            messagebox.showerror("Error", "Please select a model")
            return

        # Use processed image if available, otherwise original
        image_to_process = self.processed_image if self.processed_image else self.original_image

        if not image_to_process:
            messagebox.showerror("Error", "No image to process")
            return

        def ocr_thread():
            try:
                self.status_var.set("Performing OCR...")
                self.ocr_btn.configure(state='disabled')

                # Save image temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image_to_process.save(tmp.name, 'PNG')
                    temp_path = tmp.name

                try:
                    model_name = self.model_var.get()

                    # Parse model type and name
                    if ':' in model_name:
                        model_type, model_id = model_name.split(':', 1)
                    else:
                        model_type, model_id = 'system', model_name

                    # Build command based on model type
                    if model_type == 'custom':
                        tessdata_dir = str(self.models_dir)
                        cmd = ['tesseract', temp_path, 'stdout', '-l', model_id, '--tessdata-dir', tessdata_dir]
                    elif model_type == 'tessdata':
                        tessdata_dir = str(self.tessdata_dir)
                        cmd = ['tesseract', temp_path, 'stdout', '-l', model_id, '--tessdata-dir', tessdata_dir]
                    else:  # system
                        cmd = ['tesseract', temp_path, 'stdout', '-l', model_id]

                    # Add OCR engine mode for better accuracy
                    cmd.extend(['--oem', '1', '--psm', '6'])

                    # Run OCR
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        extracted_text = result.stdout.strip()
                        self.root.after(0, self.update_ocr_results, extracted_text)
                    else:
                        error_msg = result.stderr or "OCR failed"
                        self.root.after(0, self.show_ocr_error, error_msg)

                finally:
                    os.unlink(temp_path)  # Clean up temp file

            except Exception as e:
                self.root.after(0, self.show_ocr_error, str(e))
            finally:
                self.root.after(0, lambda: self.ocr_btn.configure(state='normal'))

        threading.Thread(target=ocr_thread, daemon=True).start()

    def update_ocr_results(self, text):
        """Update the OCR results in the UI"""
        self.text_result.delete(1.0, tk.END)
        self.text_result.insert(1.0, text)

        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))

        self.status_var.set(f"OCR complete: {line_count} lines, {word_count} words, {char_count} characters")

    def show_ocr_error(self, error):
        """Show OCR error"""
        self.text_result.delete(1.0, tk.END)
        self.text_result.insert(1.0, f"OCR Error: {error}")
        self.status_var.set("OCR failed")

    def save_text(self):
        """Save extracted text to file"""
        text = self.text_result.get(1.0, tk.END).strip()
        if not text or text.startswith("OCR Error"):
            messagebox.showwarning("Warning", "No valid text to save")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Extracted Text",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("UTF-8 Text", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_var.set(f"Text saved to: {Path(filename).name}")
                messagebox.showinfo("Success", f"Text saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save text:\n{e}")

    def clear_results(self):
        """Clear all results"""
        self.text_result.delete(1.0, tk.END)
        self.original_label.configure(image="", text="No image selected")
        self.processed_label.configure(image="", text="No processed image")
        self.original_label.image = None
        self.processed_label.image = None
        self.image_var.set("")
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.status_var.set("Ready")
        self.check_ready_state()

    def select_batch_folder(self):
        """Select folder for batch processing"""
        folder = filedialog.askdirectory(title="Select folder containing images")
        if folder:
            self.batch_folder = folder
            self.status_var.set(f"Selected batch folder: {Path(folder).name}")

    def process_batch(self):
        """Process all images in selected folder"""
        if not hasattr(self, 'batch_folder'):
            messagebox.showerror("Error", "Please select a folder first")
            return

        if not self.model_var.get():
            messagebox.showerror("Error", "Please select a model first")
            return

        def batch_thread():
            try:
                folder_path = Path(self.batch_folder)
                image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp'}

                image_files = []
                for ext in image_extensions:
                    image_files.extend(folder_path.glob(f'*{ext}'))
                    image_files.extend(folder_path.glob(f'*{ext.upper()}'))

                if not image_files:
                    self.root.after(0, lambda: messagebox.showwarning("Warning", "No image files found in selected folder"))
                    return

                self.root.after(0, lambda: self.batch_results.delete(1.0, tk.END))
                self.root.after(0, lambda: self.batch_results.insert(tk.END, f"Processing {len(image_files)} images...\n\n"))

                for i, image_file in enumerate(image_files):
                    try:
                        # Load and process image
                        image = Image.open(image_file)
                        if image.mode in ('RGBA', 'LA', 'P'):
                            image = image.convert('RGB')

                        # Apply basic enhancement
                        enhanced = ImageOps.autocontrast(image.convert('L'), cutoff=2)

                        # Save temporarily and run OCR
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            enhanced.save(tmp.name, 'PNG')
                            temp_path = tmp.name

                        try:
                            model_name = self.model_var.get()
                            if ':' in model_name:
                                model_type, model_id = model_name.split(':', 1)
                            else:
                                model_type, model_id = 'system', model_name

                            if model_type == 'tessdata':
                                tessdata_dir = str(self.tessdata_dir)
                                cmd = ['tesseract', temp_path, 'stdout', '-l', model_id, '--tessdata-dir', tessdata_dir]
                            else:
                                cmd = ['tesseract', temp_path, 'stdout', '-l', model_id]

                            result = subprocess.run(cmd, capture_output=True, text=True)

                            if result.returncode == 0:
                                text = result.stdout.strip()
                                # Save text file
                                text_file = image_file.with_suffix('.txt')
                                with open(text_file, 'w', encoding='utf-8') as f:
                                    f.write(text)

                                self.root.after(0, lambda f=image_file.name, t=len(text):
                                               self.batch_results.insert(tk.END, f"✅ {f}: {t} characters\n"))
                            else:
                                self.root.after(0, lambda f=image_file.name:
                                               self.batch_results.insert(tk.END, f"❌ {f}: OCR failed\n"))

                        finally:
                            os.unlink(temp_path)

                    except Exception as e:
                        self.root.after(0, lambda f=image_file.name, err=str(e):
                                       self.batch_results.insert(tk.END, f"❌ {f}: Error - {err}\n"))

                    # Update progress
                    progress = (i + 1) / len(image_files) * 100
                    self.root.after(0, lambda p=progress: self.status_var.set(f"Batch processing: {p:.1f}% complete"))

                self.root.after(0, lambda: self.batch_results.insert(tk.END, f"\n✅ Batch processing complete!"))
                self.root.after(0, lambda: self.status_var.set("Batch processing complete"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Batch Error", f"Batch processing failed: {e}"))

        threading.Thread(target=batch_thread, daemon=True).start()


def main():
    """Main function to run the enhanced application"""
    # Check for required packages
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Installing required packages...")
        subprocess.run(['pip3', 'install', 'opencv-python', 'numpy', '--break-system-packages'], check=True)
        import cv2
        import numpy as np

    root = tk.Tk()
    app = EnhancedOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
