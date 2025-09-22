import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import xgboost  
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("surface_roughness_model.pkl")
scaler = joblib.load("scaler.pkl")


feature_names = ["Cutting Speed", "Feed Rate", "Depth of Cut", "Tool Wear"]

entries = {}

def predict_surface_roughness():
    try:
        values = []
        for name in feature_names:
            val = float(entries[name].get())
            values.append(val)

        X_new = np.array([values])
        X_scaled = scaler.transform(X_new)
        y_pred = model.predict(X_scaled)

        messagebox.showinfo("Prediction", f"Predicted Surface Roughness (Ra): {y_pred[0]:.4f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# GUI setup 
root = tk.Tk()
root.title("Surface Roughness Predictor")


for i, name in enumerate(feature_names):
    label = tk.Label(root, text=name + ":")
    label.grid(row=i, column=0, padx=5, pady=5, sticky="e")

    entry = tk.Entry(root, width=20)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries[name] = entry

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_surface_roughness)
predict_button.grid(row=len(feature_names), column=0, columnspan=2, pady=10)

root.mainloop()
