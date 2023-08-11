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

# from models.model_test import model2
from models.ResNet18_v1 import model2

# print(os.system('pwd'))

model_path = '../models/cat_dog_resnet18_weights.pt'
# model_path = 'cat_dog_resnet18_weights.pt'

model2.load_state_dict(torch.load(model_path, map_location='cpu'))
# model2 = model2.to(device='cpu')


# checkpoint = torch.load('models/catcpu_dog_resnet18_weights.pt')
# model2.load_state_dict(checkpoint)

# функции
# Функция для загрузки изображения из URL
def load_image_from_url(url):
    response = requests.get(url)
    img_data = BytesIO(response.content)
    return Image.open(img_data)

# # Получение класса изображения
# def predict_image_class(url):
#     with torch.no_grad():
#         image = load_image_from_url(url)
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
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    dict = {0:'cat', 1:'dog'}
    output = model2(image_tensor).sigmoid().round() 
    st.write(f"Predicted class: {dict[output.item()]}")
else:
    st.write("Пожалуйста, загрузите изображение или предоставьте URL.")
