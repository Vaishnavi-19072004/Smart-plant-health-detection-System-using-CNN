import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ MODEL PATH
model = load_model("d:/plant health detection system/plant_project/plant_project/model/cnn_model.h5")

logo_path = "d:/plant health detection system/plant_project/plant_project/logo.png"

# Class labels
class_names = ["Healthy", "Unhealthy"]

# Tips dictionary
tips = {
    "Healthy": "✅ Plant is healthy.\n\nTips:\n• Continue regular watering\n• Ensure enough sunlight\n• Use organic fertilizer monthly",
    "Unhealthy": "⚠️ Plant is unhealthy.\n\nTips:\n• Remove infected leaves\n• Avoid over-watering\n• Use recommended pesticide\n• Ensure proper air circulation"
}

# Predict function
def predict_image():
    file_path = filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if file_path == "":
        return

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    result_index = np.argmax(prediction)
    result = class_names[result_index]

    # Update UI
    result_label.config(
        text=f"Result: {result}",
        fg="green" if result == "Healthy" else "red"
    )

    percent_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )

    tips_text.delete("1.0", tk.END)
    tips_text.insert(tk.END, tips[result])

# Tkinter window
root = tk.Tk()

# ✅ SAFE LOGO LOAD (error varama irukka)
try:
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((120, 120))
    logo = ImageTk.PhotoImage(logo_img)

    logo_label = tk.Label(root, image=logo)
    logo_label.pack(pady=10)

    root.iconphoto(False, tk.PhotoImage(file=logo_path))
except:
    print("⚠️ Logo not found, skipping...")

root.title("🌱 Smart Plant Health Detection")
root.geometry("500x500")
root.configure(bg="#e8f5e9")

# Title
title = tk.Label(
    root,
    text="Smart Plant Health Detection",
    font=("Helvetica", 18, "bold"),
    bg="#e8f5e9",
    fg="#1b5e20"
)
title.pack(pady=15)

# Button
upload_btn = tk.Button(
    root,
    text="Upload Leaf Image",
    command=predict_image,
    font=("Arial", 12),
    bg="#4caf50",
    fg="white",
    padx=10,
    pady=5
)
upload_btn.pack(pady=10)

# Result label
result_label = tk.Label(
    root,
    text="Result: ",
    font=("Arial", 14, "bold"),
    bg="#e8f5e9"
)
result_label.pack(pady=5)

# Percentage label
percent_label = tk.Label(
    root,
    text="Confidence: ",
    font=("Arial", 12),
    bg="#e8f5e9"
)
percent_label.pack(pady=5)

# Tips box
tips_title = tk.Label(
    root,
    text="Plant Care Tips",
    font=("Arial", 14, "bold"),
    bg="#e8f5e9"
)
tips_title.pack(pady=10)

tips_text = tk.Text(
    root,
    height=8,
    width=50,
    font=("Arial", 11)
)
tips_text.pack(pady=5)

root.mainloop()