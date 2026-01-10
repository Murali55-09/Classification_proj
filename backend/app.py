from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "../frontend/templates")
STATIC_DIR = os.path.join(BASE_DIR, "../frontend/static")


app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# 🔹 Load model safely
model_path = os.path.join(BASE_DIR, "model","cattle_breed_model.keras")
model = tf.keras.models.load_model(model_path)  

# Load class labels
with open(os.path.join(BASE_DIR, "class_indices.json"), "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return jsonify({
        "breed": labels[predicted_class],
        "confidence": float(np.max(prediction)) * 100
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
