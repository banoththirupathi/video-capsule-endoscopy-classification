from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import uuid

MODEL_PATH = "best_model.pth"
UPLOAD_FOLDER = "static/uploads"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

model = load_model(MODEL_PATH)

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[0][predicted_idx].item()

    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", error="No file uploaded.")
        
        file = request.files['image']
        if file.filename == '':
            return render_template("index.html", error="No file selected.")
        
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence = predict_image(filepath)
        return render_template("index.html", 
                               prediction=predicted_class, 
                               confidence=confidence, 
                               image_url=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
