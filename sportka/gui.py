"""
User-friendly GUI for Sportka Predictor.

Features:
- Load data
- Control learning process
- View predictions
- Store selected numbers
- Print numbers
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import json
from datetime import datetime
from typing import Optional
import numpy as np


class SportkaGUI:
    """Main GUI application for Sportka Predictor."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Sportka Predictor - Advanced ML System")
        self.root.geometry("1000x800")
        
        # State variables
        self.draw_history = None
        self.predictor = None
        self.training_thread = None
        self.training_active = False
        self.selected_numbers = []
        self.message_queue = queue.Queue()
        
        # Setup UI
        self.setup_ui()
        
        # Start message processing
        self.process_messages()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_prediction_tab()
        self.setup_results_tab()
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_data_tab(self):
        """Setup the data loading tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Data")
        
        # Data loading section
        data_frame = ttk.LabelFrame(frame, text="Data Loading", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Download button
        ttk.Button(
            data_frame,
            text="Download Latest Data from Sazka",
            command=self.download_data
        ).pack(pady=5)
        
        # Load from file
        load_frame = ttk.Frame(data_frame)
        load_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(load_frame, text="Or load from file:").pack(side=tk.LEFT, padx=5)
        
        self.data_file_var = tk.StringVar(value="/tmp/sportka.csv")
        ttk.Entry(
            load_frame,
            textvariable=self.data_file_var,
            width=40
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            load_frame,
            text="Browse",
            command=self.browse_data_file
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            load_frame,
            text="Load",
            command=self.load_data
        ).pack(side=tk.LEFT, padx=5)
        
        # Data info
        self.data_info_text = scrolledtext.ScrolledText(
            data_frame,
            height=15,
            state=tk.DISABLED
        )
        self.data_info_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def setup_training_tab(self):
        """Setup the training control tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Training")
        
        # Configuration section
        config_frame = ttk.LabelFrame(frame, text="Training Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Epochs
        epoch_frame = ttk.Frame(config_frame)
        epoch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT, padx=5)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(
            epoch_frame,
            from_=10,
            to=1000,
            textvariable=self.epochs_var,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Batch size
        batch_frame = ttk.Frame(config_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT, padx=5)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(
            batch_frame,
            from_=8,
            to=256,
            textvariable=self.batch_size_var,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Use biquaternion
        self.use_biquaternion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_frame,
            text="Use Biquaternion Transformation (Advanced)",
            variable=self.use_biquaternion_var
        ).pack(pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(pady=10)
        
        self.train_button = ttk.Button(
            control_frame,
            text="Start Training",
            command=self.start_training
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self.stop_training,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.training_log = scrolledtext.ScrolledText(
            progress_frame,
            height=15,
            state=tk.DISABLED
        )
        self.training_log.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Model management
        model_frame = ttk.LabelFrame(frame, text="Model Management", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        model_button_frame = ttk.Frame(model_frame)
        model_button_frame.pack(pady=5)
        
        ttk.Button(
            model_button_frame,
            text="Save Model",
            command=self.save_model
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            model_button_frame,
            text="Load Model",
            command=self.load_model
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            model_button_frame,
            text="List Saved Models",
            command=self.list_models
        ).pack(side=tk.LEFT, padx=5)
    
    def setup_prediction_tab(self):
        """Setup the prediction tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Prediction")
        
        # Prediction control
        control_frame = ttk.LabelFrame(frame, text="Make Prediction", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        date_frame = ttk.Frame(control_frame)
        date_frame.pack(pady=5)
        
        ttk.Label(date_frame, text="Prediction Date (DD.MM.YYYY):").pack(side=tk.LEFT, padx=5)
        self.predict_date_var = tk.StringVar(value=datetime.now().strftime("%d.%m.%Y"))
        ttk.Entry(
            date_frame,
            textvariable=self.predict_date_var,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            date_frame,
            text="Predict",
            command=self.make_prediction
        ).pack(side=tk.LEFT, padx=5)
        
        # Prediction results
        results_frame = ttk.LabelFrame(frame, text="Predictions", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.prediction_text = scrolledtext.ScrolledText(
            results_frame,
            height=20,
            font=('Courier', 12)
        )
        self.prediction_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def setup_results_tab(self):
        """Setup the results and printing tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Results")
        
        # Selected numbers
        selected_frame = ttk.LabelFrame(frame, text="Selected Numbers", padding=10)
        selected_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.selected_listbox = tk.Listbox(selected_frame, height=10, font=('Courier', 11))
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        button_frame = ttk.Frame(selected_frame)
        button_frame.pack(pady=5)
        
        ttk.Button(
            button_frame,
            text="Add Current Prediction",
            command=self.add_prediction_to_selected
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Selection",
            command=self.clear_selected
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Save to File",
            command=self.save_selected
        ).pack(side=tk.LEFT, padx=5)
        
        # Printing
        print_frame = ttk.LabelFrame(frame, text="Print", padding=10)
        print_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            print_frame,
            text="Print Selected Numbers",
            command=self.print_numbers
        ).pack(pady=5)
        
        ttk.Label(
            print_frame,
            text="(Generates PDF for printing)"
        ).pack(pady=5)
    
    # Event handlers
    
    def download_data(self):
        """Download latest data from Sazka."""
        self.log_message("Downloading data from Sazka...")
        try:
            from sportka.download import download_data_from_sazka
            download_data_from_sazka()
            self.log_message("Data downloaded successfully!")
            self.load_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download data: {str(e)}")
            self.log_message(f"Error downloading data: {str(e)}")
    
    def browse_data_file(self):
        """Browse for data file."""
        filename = filedialog.askopenfilename(
            title="Select Sportka Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_file_var.set(filename)
    
    def load_data(self):
        """Load data from file."""
        self.log_message("Loading data...")
        try:
            from sportka.learn import draw_history
            
            # Set the data file path if custom
            data_file = self.data_file_var.get()
            if os.path.exists(data_file):
                # Temporarily modify download function or copy file
                import shutil
                if data_file != '/tmp/sportka.csv':
                    shutil.copy(data_file, '/tmp/sportka.csv')
            
            self.draw_history = draw_history()
            
            # Display info
            self.data_info_text.config(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, f"Loaded {len(self.draw_history.draws)} draws\n\n")
            self.data_info_text.insert(tk.END, "Last 5 draws:\n")
            for draw in self.draw_history.draws[-5:]:
                self.data_info_text.insert(
                    tk.END,
                    f"{draw.date}: {draw.first} / {draw.second}\n"
                )
            self.data_info_text.config(state=tk.DISABLED)
            
            self.log_message(f"Loaded {len(self.draw_history.draws)} draws successfully!")
            self.status_bar.config(text=f"Data loaded: {len(self.draw_history.draws)} draws")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.log_message(f"Error loading data: {str(e)}")
    
    def start_training(self):
        """Start training in background thread."""
        if self.draw_history is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        if self.training_active:
            messagebox.showwarning("Warning", "Training already in progress!")
            return
        
        self.training_active = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.train_model, daemon=True)
        self.training_thread.start()
    
    def train_model(self):
        """Train the model (runs in background thread)."""
        try:
            from sportka.neural_network import SportkaPredictor, create_training_data_with_biquaternion
            
            self.log_training("Initializing predictor...")
            
            # Determine input dimension based on biquaternion usage
            use_bq = self.use_biquaternion_var.get()
            input_dim = 103 if not use_bq else 103 + 48  # Base + 2*24 biquaternion features
            
            self.predictor = SportkaPredictor(
                input_dim=input_dim,
                hidden_layers=32,
                hidden_units=128,
                dropout_rate=0.3
            )
            
            self.log_training("Building model...")
            self.predictor.build_model()
            
            self.log_training("Preparing training data...")
            x_train, y_train = create_training_data_with_biquaternion(
                self.draw_history.draws,
                use_biquaternion=use_bq
            )
            
            self.log_training(f"Training data shape: {x_train.shape}")
            self.log_training(f"Starting training for {self.epochs_var.get()} epochs...")
            
            # Custom callback to update GUI
            class GUICallback(tf.keras.callbacks.Callback):
                def __init__(self, gui, total_epochs):
                    super().__init__()
                    self.gui = gui
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = ((epoch + 1) / self.total_epochs) * 100
                    self.gui.message_queue.put(('progress', progress))
                    
                    log_msg = f"Epoch {epoch+1}/{self.total_epochs} - "
                    log_msg += f"loss: {logs.get('loss', 0):.4f} - "
                    log_msg += f"val_loss: {logs.get('val_loss', 0):.4f}"
                    self.gui.message_queue.put(('log', log_msg))
            
            import tensorflow as tf
            
            # Train
            history = self.predictor.train(
                x_train, y_train,
                epochs=self.epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                validation_split=0.2,
                callbacks=[GUICallback(self, self.epochs_var.get())],
                verbose=0
            )
            
            self.log_training("Training completed!")
            self.message_queue.put(('training_done', None))
            
        except Exception as e:
            self.message_queue.put(('error', f"Training error: {str(e)}"))
        finally:
            self.training_active = False
    
    def stop_training(self):
        """Stop training."""
        self.training_active = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_training("Training stopped by user")
    
    def save_model(self):
        """Save the trained model."""
        if self.predictor is None:
            messagebox.showwarning("Warning", "No model to save!")
            return
        
        try:
            path = self.predictor.save_weights()
            messagebox.showinfo("Success", f"Model saved to:\n{path}")
            self.log_message(f"Model saved: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load a saved model."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            initialdir="./models"
        )
        
        if filename:
            try:
                from sportka.neural_network import SportkaPredictor
                
                self.predictor = SportkaPredictor()
                self.predictor.load_weights(filename)
                
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.log_message(f"Model loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def list_models(self):
        """List all saved models."""
        from sportka.neural_network import SportkaPredictor
        
        predictor = SportkaPredictor()
        models = predictor.list_saved_models()
        
        if not models:
            messagebox.showinfo("Saved Models", "No saved models found")
            return
        
        model_list = "\n".join([os.path.basename(m) for m in models])
        messagebox.showinfo("Saved Models", f"Available models:\n\n{model_list}")
    
    def make_prediction(self):
        """Make a prediction for the specified date."""
        if self.predictor is None:
            messagebox.showwarning("Warning", "Please train or load a model first!")
            return
        
        if self.draw_history is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        try:
            from datetime import datetime
            from sportka.learn import date_to_x
            
            # Parse date
            date_str = self.predict_date_var.get()
            predict_date = datetime.strptime(date_str, '%d.%m.%Y').date()
            
            # Prepare input
            x_predict = date_to_x(predict_date)
            x_hist1 = self.draw_history.draws[-1].y_train_1
            x_hist2 = self.draw_history.draws[-1].y_train_2
            
            if self.use_biquaternion_var.get():
                from sportka.biquaternion import apply_biquaternion_theta_transform
                
                # Get top numbers and transform
                top_nums_1 = np.argsort(x_hist1)[-7:] + 1
                top_nums_2 = np.argsort(x_hist2)[-7:] + 1
                
                bq1 = apply_biquaternion_theta_transform(top_nums_1.tolist())
                bq2 = apply_biquaternion_theta_transform(top_nums_2.tolist())
                
                x_full = np.concatenate([x_predict, x_hist1, x_hist2, bq1, bq2])
            else:
                x_full = np.concatenate([x_predict, x_hist1, x_hist2])
            
            x_input = np.array([x_full])
            
            # Make prediction
            prediction = self.predictor.predict(x_input)
            best_numbers = self.predictor.get_best_numbers(prediction, n=7)
            
            # Display results
            self.prediction_text.delete(1.0, tk.END)
            self.prediction_text.insert(tk.END, f"Prediction for {date_str}\n")
            self.prediction_text.insert(tk.END, "=" * 50 + "\n\n")
            self.prediction_text.insert(tk.END, "Top 7 Numbers (with probabilities):\n\n")
            
            for i, (num, prob) in enumerate(best_numbers, 1):
                self.prediction_text.insert(
                    tk.END,
                    f"{i}. Number {num:2d} - Probability: {prob:.4f}\n"
                )
            
            self.prediction_text.insert(tk.END, "\n" + "=" * 50 + "\n")
            self.prediction_text.insert(
                tk.END,
                f"\nRecommended numbers: {[n for n, _ in best_numbers]}\n"
            )
            
            # Store for later use
            self.current_prediction = best_numbers
            self.current_prediction_date = date_str
            
            self.log_message(f"Prediction made for {date_str}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.log_message(f"Prediction error: {str(e)}")
    
    def add_prediction_to_selected(self):
        """Add current prediction to selected numbers."""
        if not hasattr(self, 'current_prediction'):
            messagebox.showwarning("Warning", "No prediction available!")
            return
        
        numbers = [n for n, _ in self.current_prediction]
        date = self.current_prediction_date
        
        self.selected_numbers.append({
            'date': date,
            'numbers': numbers,
            'timestamp': datetime.now().isoformat()
        })
        
        self.selected_listbox.insert(
            tk.END,
            f"{date}: {numbers}"
        )
        
        self.log_message(f"Added prediction to selected numbers")
    
    def clear_selected(self):
        """Clear selected numbers."""
        self.selected_numbers = []
        self.selected_listbox.delete(0, tk.END)
        self.log_message("Cleared selected numbers")
    
    def save_selected(self):
        """Save selected numbers to file."""
        if not self.selected_numbers:
            messagebox.showwarning("Warning", "No numbers selected!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Selected Numbers",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.selected_numbers, f, indent=2)
                messagebox.showinfo("Success", f"Saved to {filename}")
                self.log_message(f"Saved selected numbers to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def print_numbers(self):
        """Print selected numbers to PDF."""
        if not self.selected_numbers:
            messagebox.showwarning("Warning", "No numbers selected!")
            return
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            
            filename = filedialog.asksaveasfilename(
                title="Save PDF",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
            
            if not filename:
                return
            
            # Create PDF
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            
            # Title
            c.setFont("Helvetica-Bold", 20)
            c.drawString(inch, height - inch, "Sportka Predictions")
            
            # Numbers
            c.setFont("Helvetica", 14)
            y = height - 1.5 * inch
            
            for i, entry in enumerate(self.selected_numbers):
                if y < inch:  # New page if needed
                    c.showPage()
                    y = height - inch
                
                c.drawString(inch, y, f"{entry['date']}: {entry['numbers']}")
                y -= 0.3 * inch
            
            c.save()
            
            messagebox.showinfo("Success", f"PDF saved to {filename}")
            self.log_message(f"Printed numbers to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create PDF: {str(e)}")
    
    # Helper methods
    
    def log_message(self, message: str):
        """Log a message to status."""
        self.status_bar.config(text=message)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def log_training(self, message: str):
        """Log a training message."""
        self.message_queue.put(('log', message))
    
    def log_data(self, message: str):
        """Log a data message."""
        self.data_info_text.config(state=tk.NORMAL)
        self.data_info_text.insert(tk.END, message + "\n")
        self.data_info_text.see(tk.END)
        self.data_info_text.config(state=tk.DISABLED)
    
    def process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == 'log':
                    self.training_log.config(state=tk.NORMAL)
                    self.training_log.insert(tk.END, msg_data + "\n")
                    self.training_log.see(tk.END)
                    self.training_log.config(state=tk.DISABLED)
                
                elif msg_type == 'progress':
                    self.progress_var.set(msg_data)
                
                elif msg_type == 'training_done':
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    messagebox.showinfo("Success", "Training completed!")
                
                elif msg_type == 'error':
                    messagebox.showerror("Error", msg_data)
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_messages)


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = SportkaGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
