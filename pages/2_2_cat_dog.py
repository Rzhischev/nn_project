import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision import models

import torch

import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import requests
from io import BytesIO

#модели
from models.model_test import model2
from torchvision.models import inception_v3

model = inception_v3(pretrained=True)
model2.load_state_dict(torch.load('models/model-test-cat_dog.pt', map_location=torch.device('cpu')))
    
# функции
# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)

# Получение класса изображения
def predict_image_class(url):
    with torch.no_grad():
        image = load_image_from_url(url)
        outputs = model(image)
        _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
        class_name = labels[predicted_class.item()]
        return class_name

# # image = load_image_from_url(url)
# # Получение класса изображения
# def predict_image_class(image):
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
#         class_name = labels[predicted_class.item()]
#         return class_name

st.title("Классификация изображений котов и собак")

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
    model2.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)
    dict = {0:'cat', 1:'dog'}
    output = model2(image_tensor).sigmoid().round()
    st.write(f"Predicted class: {dict[output.item()]}")
else:
    st.write("Пожалуйста, загрузите изображение или предоставьте URL.")
