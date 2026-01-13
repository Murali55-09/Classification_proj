from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
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

# Enable CORS
CORS(app)

# 🔹 Global variables for lazy loading
interpreter = None
input_details = None
output_details = None
class_indices = None
labels = None

def load_model_lazy():
    """Load TFLite model only when first needed to save memory"""
    global interpreter, input_details, output_details, class_indices, labels
    
    if interpreter is None:
        print("🔄 Loading TFLite model for the first time...")
        try:
            import tensorflow as tf
            
            # Load TFLite model
            model_path = os.path.join(BASE_DIR, "cattle_breed_model.tflite")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Load class labels
            with open(os.path.join(BASE_DIR, "class_indices.json"), "r") as f:
                class_indices = json.load(f)
            
            labels = {v: k for k, v in class_indices.items()}
            print("✅ TFLite model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise e
    
    return interpreter, input_details, output_details, labels

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    """Render home page"""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "cattle-breed-classifier-lite"})

@app.route("/predict", methods=["POST"])
def predict():
    """Predict cattle breed from uploaded image"""
    try:
        # Check if file is present
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        
        # Check if file is empty
        if file.filename == "":
            return jsonify({"error": "Empty file uploaded"}), 400

        # Load model lazily
        interpreter, input_details, output_details, labels = load_model_lazy()

        # Open and preprocess image
        image = Image.open(file).convert("RGB")
        img = preprocess_image(image)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100

        return jsonify({
            "breed": labels[predicted_class],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask app with TFLite on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)