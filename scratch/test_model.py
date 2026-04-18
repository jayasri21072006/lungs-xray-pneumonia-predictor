import torch
from xray.ml.model.arch import Net
import os

MODEL_PATH = 'notebook/best_model.pth'

def test_model():
    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cpu")
    model = Net()
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("state_dict loaded.")
        except:
            model = torch.load(MODEL_PATH, map_location=device)
            print("Full model loaded.")
    else:
        print("Model file not found.")
        return

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output: {output}")
        
if __name__ == "__main__":
    test_model()
