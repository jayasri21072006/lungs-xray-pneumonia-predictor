import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from xray.ml.model.arch import Net
import os
import time

# Configuration
MODEL_PATH = 'notebook/best_model.pth'
DATA_PATH = r'D:\data\test'
BATCH_SIZE = 32

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load Model
    model = Net().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model state_dict loaded.")
        except:
            model = torch.load(MODEL_PATH, map_location=device)
            print("Full model loaded.")
    else:
        print("Model file not found!")
        return

    model.eval()

    # Data Transformation (matching app.py exactly)
    preprocess = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = datasets.ImageFolder(root=DATA_PATH, transform=preprocess)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    correct = 0
    total = 0
    start_time = time.time()

    print(f"Starting evaluation on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Handling 2 classes output
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 5 == 0:
                print(f"Processed {total}/{len(test_dataset)} images...")

    end_time = time.time()
    accuracy = 100 * correct / total
    
    print("\n" + "="*30)
    print(f"Evaluation Results:")
    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print("="*30)

if __name__ == "__main__":
    evaluate()
