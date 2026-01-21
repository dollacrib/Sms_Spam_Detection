import os, sys, pickle, re

# Auto-install missing packages (runs only the first time)
try:
    import nltk
except:
    print("Installing nltk...")
    os.system(f"{sys.executable} -m pip install nltk --quiet")
    import nltk

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except:
    print("Installing scikit-learn...")
    os.system(f"{sys.executable} -m pip install scikit-learn --quiet")
    from sklearn.feature_extraction.text import TfidfVectorizer

import tkinter as tk
from tkinter import messagebox
from nltk.corpus import stopwords

print("Downloading stopwords (only once)...")
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(w for w in text.split() if w not in stop)

# Load model & vectorizer
try:
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    messagebox.showerror("Missing Files!", "Make sure these 3 files are in the same folder:\n• spam_model.pkl\n• vectorizer.pkl\n• spam_gui.py")
    input("Press Enter to close...")
    sys.exit()

def check():
    msg = box.get("1.0", "end-1c").strip()
    if not msg:
        messagebox.showwarning("Empty", "Please type a message!")
        return
    cleaned = clean(msg)
    pred = model.predict(vectorizer.transform([cleaned]))[0]
    prob = model.predict_proba(vectorizer.transform([cleaned]))[0].max()
    result = "SPAM" if pred else "NOT SPAM"
    color = "red" if pred else "green"
    label.config(text=f"{result}\nConfidence: {prob:.1%}", foreground=color)

# GUI
root = tk.Tk()
root.title("Spam Detector AI")
root.geometry("750x650")
root.configure(bg="#f0f2f5")

tk.Label(root, text="Spam Message Detector", font=("Arial", 24, "bold"), bg="#f0f2f5", fg="#2c3e50").pack(pady=30)
tk.Label(root, text="Enter your message below:", font=("Arial", 13), bg="#f0f2f5").pack()

box = tk.Text(root, height=10, width=80, font=("Arial", 12))
box.pack(pady=20)

tk.Button(root, text="Check for Spam", command=check, font=("Arial", 14), bg="#e74c3c", fg="white", height=2, width=20).pack(pady=15)

label = tk.Label(root, text="Result will appear here", font=("Arial", 20, "bold"), bg="#f0f2f5", fg="gray")
label.pack(pady=30)

print("Spam Detector is ready! Opening window...")
root.mainloop()