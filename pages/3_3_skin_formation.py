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

import streamlit as st
from PIL import Image
import requests
from io import BytesIO

#модели
from models.model_test import model2
from torchvision.models import inception_v3

model = inception_v3(pretrained=True)
model2.load_state_dict(torch.load('models/model-test-cat_dog.pt', map_location=torch.device('cpu')))

def show():
    skin_formation_classifier()


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

# image = load_image_from_url(url)
# Получение класса изображения
def predict_image_class(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
        class_name = labels[predicted_class.item()]
        return class_name

st.title("Классификация образований на коже")

# uploaded_file = st.file_uploader("Загрузите изображение кожи...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
    
#     # Загрузите ваш код для классификации образований на коже