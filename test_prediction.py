import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from xray.ml.model.arch import Net

def test_model():
    # Configuration
    MODEL_PATH = 'notebook/best_model.pth'
    TEST_DIR = 'test_images'
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model state_dict loaded successfully.")
        except Exception as e:
            try:
                model = torch.load(MODEL_PATH, map_location=device)
                print("Full model loaded successfully.")
            except:
                print(f"Error loading model: {e}")
                return
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return
    
    model.eval()
    
    # Image Transformation
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class_names = ["NORMAL", "PNEUMONIA"]
    
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print("\n--- Testing Model Predictions ---")
    for filename in test_files:
        img_path = os.path.join(TEST_DIR, filename)
        image = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_batch)
            # Use softmax if the model doesn't already have it
            # Based on app.py, it seems to use raw logits or sigmoid/softmax already?
            # Let's check the Net architecture if possible, but app.py just uses torch.max(output, 1)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        result = class_names[predicted.item()]
        score = confidence.item() * 100
        
        print(f"File: {filename}")
        print(f"Prediction: {result}")
        print(f"Confidence: {score:.2f}%")
        print(f"Logits: {output[0].tolist()}")
        print("-" * 30)

if __name__ == '__main__':
    test_model()
