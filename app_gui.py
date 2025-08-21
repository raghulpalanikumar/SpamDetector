import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import spam_detector as sd


class SpamDetectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Email/SMS Spam Detector")
        self.geometry("720x600")
        self.configure(bg="#f7f7fa")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f7f7fa")
        style.configure("TLabel", background="#f7f7fa", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#2a2a2a", background="#f7f7fa")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"), background="#f7f7fa")
        style.configure("Pred.TLabel", font=("Segoe UI", 12, "bold"), background="#f7f7fa")
        style.configure("Footer.TLabel", font=("Segoe UI", 9), foreground="#888", background="#f7f7fa")
        style.configure("Nav.TButton", font=("Segoe UI", 10, "bold"), padding=6)

        self.model_path_var = tk.StringVar(value=sd.DEFAULT_MODEL_PATH)
        self.data_path_var = tk.StringVar(value=sd.DEFAULT_DATA_PATH)
        self.recent_predictions = []

        self._build_ui()
        self.model_loaded = False

    def _build_ui(self) -> None:
        # Navbar
        navbar = ttk.Frame(self)
        navbar.pack(fill=tk.X, padx=0, pady=0)
        ttk.Button(navbar, text="Home", style="Nav.TButton", command=self._show_home).pack(side=tk.LEFT, padx=(12, 2), pady=4)
        ttk.Button(navbar, text="About", style="Nav.TButton", command=self._show_about).pack(side=tk.LEFT, padx=2, pady=4)
        ttk.Button(navbar, text="Help", style="Nav.TButton", command=self._show_help).pack(side=tk.LEFT, padx=2, pady=4)

        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=0, pady=(0, 8))
        ttk.Label(header, text="📧 Email/SMS Spam Detector", style="Header.TLabel").pack(anchor="center", pady=(16, 0))
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=10, pady=(0, 10))

        # Paths row
        paths_frame = ttk.Frame(self)
        paths_frame.pack(fill=tk.X, padx=18, pady=8)
        ttk.Label(paths_frame, text="Model:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(paths_frame, textvariable=self.model_path_var, width=54).grid(row=0, column=1, padx=6, pady=2)
        ttk.Button(paths_frame, text="Browse", command=self._browse_model).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(paths_frame, text="Load Model", command=self._load_model).grid(row=0, column=3, padx=6, pady=2)
        ttk.Label(paths_frame, text="Data CSV:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(paths_frame, textvariable=self.data_path_var, width=54).grid(row=1, column=1, padx=6, pady=2)
        ttk.Button(paths_frame, text="Browse", command=self._browse_data).grid(row=1, column=2, padx=2, pady=2)
        ttk.Button(paths_frame, text="Train Model", command=self._train_model).grid(row=1, column=3, padx=6, pady=2)
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=10, pady=8)

        # Input + Predict
        input_frame = ttk.Frame(self)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=4)
        ttk.Label(input_frame, text="Enter message:").pack(anchor="w", pady=(0, 2))
        self.text_input = tk.Text(input_frame, height=8, wrap=tk.WORD, font=("Segoe UI", 10), bg="#fff", relief=tk.GROOVE, bd=2)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        controls = ttk.Frame(input_frame)
        controls.pack(fill=tk.X, pady=6)
        ttk.Button(controls, text="Predict", command=self._predict).pack(side=tk.LEFT)
        ttk.Button(controls, text="Clear", command=self._clear_fields).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Copy Result", command=self._copy_result).pack(side=tk.LEFT, padx=(8, 0))
        self.status_label = ttk.Label(controls, text="Status: Ready", style="Status.TLabel", foreground="#333")
        self.status_label.pack(side=tk.RIGHT)

        # Output
        output_frame = ttk.Frame(self)
        output_frame.pack(fill=tk.X, padx=18, pady=8)
        self.pred_label_var = tk.StringVar(value="Prediction: -")
        self.prob_label_var = tk.StringVar(value="Spam probability: -")
        self.pred_label = ttk.Label(output_frame, textvariable=self.pred_label_var, style="Pred.TLabel")
        self.pred_label.pack(anchor="w")
        ttk.Label(output_frame, textvariable=self.prob_label_var).pack(anchor="w")

        # Recent Predictions
        recent_frame = ttk.Frame(self)
        recent_frame.pack(fill=tk.BOTH, expand=False, padx=18, pady=(0, 8))
        ttk.Label(recent_frame, text="Recent Predictions:").pack(anchor="w")
        self.recent_listbox = tk.Listbox(recent_frame, height=4, font=("Segoe UI", 9), bg="#f9f9fc", relief=tk.GROOVE, bd=1)
        self.recent_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 2))

        # Footer
        footer = ttk.Frame(self)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(0, 0))
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=0, pady=(0, 0))
        ttk.Label(footer, text="© 2024 Email/SMS Spam Detector | Contact: support@example.com", style="Footer.TLabel").pack(anchor="center", pady=4)

    # UI helpers and new features
    def _set_status(self, text: str, color: str = "#333") -> None:
        self.status_label.config(text=f"Status: {text}", foreground=color)
        self.update_idletasks()

    def _clear_fields(self) -> None:
        self.text_input.delete("1.0", tk.END)
        self.pred_label_var.set("Prediction: -")
        self.prob_label_var.set("Spam probability: -")
        self._set_status("Cleared", "#333")

    def _copy_result(self) -> None:
        result = self.pred_label_var.get() + "\n" + self.prob_label_var.get()
        self.clipboard_clear()
        self.clipboard_append(result)
        self._set_status("Result copied", "#0a7")

    def _show_about(self):
        messagebox.showinfo("About", "Email/SMS Spam Detector\n\nThis app uses machine learning to detect spam messages.\n\nDeveloped 2024.")

    def _show_help(self):
        messagebox.showinfo("Help", "1. Load or train a model.\n2. Enter a message.\n3. Click Predict.\n4. See the result below.\n\nUse Clear to reset fields. Use Copy Result to copy the prediction.")

    def _show_home(self):
        self._set_status("Ready", "#333")

    def _add_recent_prediction(self, text, label, prob):
        entry = f"{label.upper()} ({prob:.2f}) - {text[:40].replace('\n',' ')}{'...' if len(text)>40 else ''}"
        self.recent_predictions.insert(0, entry)
        self.recent_predictions = self.recent_predictions[:10]
        self.recent_listbox.delete(0, tk.END)
        for item in self.recent_predictions:
            self.recent_listbox.insert(tk.END, item)

    def _browse_model(self) -> None:
        path = filedialog.asksaveasfilename(title="Select/Save Model File", defaultextension=".joblib", filetypes=[("Joblib", "*.joblib"), ("All", "*.*")])
        if path:
            self.model_path_var.set(path)

    def _browse_data(self) -> None:
        path = filedialog.askopenfilename(title="Select Dataset CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.data_path_var.set(path)

    def _load_model(self) -> None:
        model_path = self.model_path_var.get().strip()
        try:
            _ = sd.load_model(model_path)
            self.model_loaded = True
            self._set_status("Model loaded", "#0a7")
        except Exception as exc:  # noqa: BLE001
            self.model_loaded = False
            messagebox.showerror("Load Model", f"Failed to load model:\n{exc}")
            self._set_status("Model load failed", "#c00")

    def _train_model(self) -> None:
        data_path = self.data_path_var.get().strip()
        model_path = self.model_path_var.get().strip()

        if not os.path.exists(data_path):
            messagebox.showwarning("Train Model", f"Dataset not found at:\n{data_path}")
            return

        def _run_train() -> None:
            try:
                self._set_status("Training… this may take ~10-30s", "#f90")
                result = sd.train_and_save(data_path=data_path, model_path=model_path)
                self.model_loaded = True
                self._set_status(
                    f"Trained. Acc={result.test_accuracy:.4f}, F1(spam)={result.test_f1_spam:.4f}",
                    "#0a7",
                )
                messagebox.showinfo(
                    "Training Complete",
                    f"Model saved to: {result.model_path}\nAccuracy: {result.test_accuracy:.4f}\nF1(spam): {result.test_f1_spam:.4f}",
                )
            except Exception as exc:  # noqa: BLE001
                self._set_status("Training failed", "#c00")
                messagebox.showerror("Training Failed", str(exc))

        threading.Thread(target=_run_train, daemon=True).start()

    def _predict(self) -> None:
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Predict", "Please enter a message.")
            return
        model_path = self.model_path_var.get().strip()
        try:
            label, prob = sd.predict_text(text, model_path=model_path)
            self.pred_label_var.set(f"Prediction: {label.upper()}")
            self.prob_label_var.set(f"Spam probability: {prob:.4f}")
            color = "#c00" if label == "spam" else "#0a7"
            self.pred_label.config(foreground=color)
            self._set_status("Predicted", "#333")
            self._add_recent_prediction(text, label, prob)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Predict Failed", str(exc))
            self._set_status("Prediction failed", "#c00")


if __name__ == "__main__":
    app = SpamDetectorApp()
    app.mainloop()

