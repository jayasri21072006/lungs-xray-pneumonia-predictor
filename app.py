import os
import sys
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from xray.ml.model.arch import Net

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'notebook/best_model.pth'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Load weights - and handle potential path issues
if os.path.exists(MODEL_PATH):
    # If the file is only state_dict
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model state_dict loaded successfully.")
    except Exception as e:
        # If the file contains the full model
        try:
            model = torch.load(MODEL_PATH, map_location=device)
            print("Full model loaded successfully.")
        except:
            print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

model.eval()

# Image Transformation
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    print(f"Received file: {file.filename}")

    try:
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # Process Image
        image = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            # Standard class order: 0=NORMAL, 1=PNEUMONIA
            prob_normal = output[0][0].item()
            prob_pneumonia = output[0][1].item()
            print(f"--- Prediction Debug ---")
            print(f"Image: {file.filename}")
            print(f"Normal Score: {prob_normal:.4f}")
            print(f"Pneumonia Score: {prob_pneumonia:.4f}")
            
            confidence, predicted = torch.max(output, 1)
            
        class_names = ["NORMAL", "PNEUMONIA"]
        result = class_names[predicted.item()]
        score = confidence.item() * 100

        return jsonify({
            'prediction': result,
            'confidence': f"{score:.2f}%",
            'image_url': f"/{img_path.replace(os.sep, '/')}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
