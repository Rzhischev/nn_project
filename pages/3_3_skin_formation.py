import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO

from models.ResNet_Cancer import model3

# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)


st.title("Классификация образований на коже")

uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])
url = st.text_input("Или введите URL изображения...")

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
elif url:
    try:
        image = load_image_from_url(url).convert("RGB")
        st.image(image, caption='Loaded Image from URL.', use_column_width=True)
    except:
        st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

if image:  # Если изображение успешно загружено
    model3.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    dict = {0:'Benign', 1:'Malignant'}
    output = model3(image_tensor).sigmoid().round() 
    st.write(f"This image likely contains a: {dict[output.item()]}")
else:
    st.write("Пожалуйста, загрузите изображение или предоставьте URL.")
