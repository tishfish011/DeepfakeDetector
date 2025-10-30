import torch
from torchvision import models, transforms
from PIL import Image
import io
import streamlit as st

# --- Load model ---
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
#model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu", weights_only=True))
#model.eval()

#Debug check
import os
print("Current working directory:", os.getcwd())
print("Files in cwd:", os.listdir())

# Load the model weights
model.load_state_dict(torch.load("main_directory/deepfake_model.pth", map_location="cpu", weights_only=True))
model.eval()
print("✅ Model weights loaded successfully")

#model.load_state_dict(torch.load("models/deepfake_model.pth", map_location="cpu", weights_only=True))
#model.eval()  # also important — puts it in inference mode


# --- Define preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    return "Real" if prediction == 0 else "Fake"

#print(classify_image("sample.jpg"))

st.title("Deepfake Detector")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    result = classify_image(uploaded)
    st.write(f"Prediction: {result}")
    

