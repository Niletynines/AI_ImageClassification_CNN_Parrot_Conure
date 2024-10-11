import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

st.write('Flower Classification CNN Model')
st.header('Flower Classification CNN Model')
classes = ['Black capped conure', 'Blue Cinnamon Greencheek', 'Blue Normal Greencheek', 'Blue Pineapple Greencheek', 'Blue Yellow sided Greencheek', 'Cinnamon Greencheek', 'Crimson bellird counre', 'Normal Greencheek', 'Pineapple Greencheek', 'Sun conure', 'Yellow sided Greencheek']

# Load PyTorch model
model = torch.load('modelNomal_100.pt', map_location=torch.device('cpu'))
model.eval() # Set the model to evaluation mode

# Image preprocessing function (for PyTorch)
def classify_images(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        predictions = model(input_tensor)
        result = torch.nn.functional.softmax(predictions[0], dim=0)
        outcome = f'The Image belongs to {classes[result.argmax()]} with a score of {result.max().item()*100:.2f}%'
    
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    image_path = os.path.join('upload', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)
    st.markdown(classify_images(image_path))
