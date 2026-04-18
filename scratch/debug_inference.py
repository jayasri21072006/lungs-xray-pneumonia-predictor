import torch
from xray.ml.model.arch import Net
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

MODEL_PATH = 'notebook/best_model.pth'
IMAGE_PATH = 'static/uploads/WhatsApp_Image_2026-04-02_at_10.56.43_PM_1.jpeg'

def debug_prediction():
    device = torch.device("cpu")
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        raw_output = model(input_batch)
        print(f"Raw Model Output (after Sigmoid in arch.py): {raw_output}")
        
        # In app.py, Softmax is applied to this
        probs = F.softmax(raw_output, dim=1)
        print(f"Probabilities (Softmax on Sigmoid): {probs}")
        
        conf, pred = torch.max(probs, 1)
        class_names = ["NORMAL", "PNEUMONIA"]
        print(f"Prediction: {class_names[pred.item()]} ({conf.item()*100:.2f}%)")

if __name__ == "__main__":
    debug_prediction()
