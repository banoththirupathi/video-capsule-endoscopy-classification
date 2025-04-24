import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

MODEL_PATH = "best_model.pth"         
IMAGE_PATH = r"Dataset\validation\Bleeding\KVASIR\d369e4f163df4aba_12018.jpg"         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[0][predicted_idx].item()

    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) > 2:
        IMAGE_PATH = sys.argv[1]
        MODEL_PATH = sys.argv[2]

    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    predicted_class, confidence = predict_image(model, IMAGE_PATH)

    print(f"\n🔍 Prediction: {predicted_class} (Confidence: {confidence:.2f})")
