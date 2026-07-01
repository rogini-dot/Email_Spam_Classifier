import tkinter as tk
from tkinter import messagebox
import pickle
import os

# Load the saved model and vectorizer
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    messagebox.showerror(
        "Error",
        f"Model files not found!\nMake sure '{MODEL_PATH}' and '{VECTORIZER_PATH}' are in the same folder."
    )
    exit()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Prediction function
def predict_email():
    email_text = text_box.get("1.0", tk.END).strip()

    if not email_text:
        messagebox.showwarning("Warning", "Please enter some email text first!")
        return

    email_vector = vectorizer.transform([email_text])
    result = model.predict(email_vector)

    if result[0] == 1:
        messagebox.showinfo("Result", "🚨 SPAM EMAIL")
    else:
        messagebox.showinfo("Result", "✅ NORMAL EMAIL")

# GUI Window
window = tk.Tk()
window.title("Spam Email Classifier")
window.geometry("500x400")

label = tk.Label(window, text="Enter Email Text:", font=("Arial", 14))
label.pack(pady=10)

text_box = tk.Text(window, height=10, width=50)
text_box.pack()

button = tk.Button(
    window,
    text="Check Email",
    command=predict_email,
    font=("Arial", 12)
)
button.pack(pady=20)

window.mainloop()