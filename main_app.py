# main_app.py
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

#модели
from models.model_test import model2
model2.load_state_dict(torch.load('models/model-test-cat_dog.pt', map_location=torch.device('cpu')))



# Функция для первой страницы
def inception_classifier():
    st.write("Классификация произвольного изображения с помощью модели Inception")

    uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)


# Функция для второй страницы
def cat_dog_classifier():
    st.write("Классификация изображений котов и собак")

    uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Загрузите ваш код для дообученной ResNet18 и классификации
        model2.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image)
        dict = {0:'cat', 1:'dog'}
        output = model2(image_tensor.unsqueeze(0)).sigmoid().round()
        st.write(f"Predicted class: {dict[output.item()]}")

# Функция для третьей страницы
def skin_formation_classifier():
    st.write("Классификация образований на коже")

    # uploaded_file = st.file_uploader("Загрузите изображение кожи...", type=["jpg", "png", "jpeg"])

    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption='Uploaded Image.', use_column_width=True)
        
    #     # Загрузите ваш код для классификации образований на коже

# Основное приложение
def main():
    st.sidebar.title("Выберите модель для классификации:")
    page = st.sidebar.radio("", ["Inception (ImageNet)", "ResNet18 (Коты и Собаки)", "Образования на коже"])

    if page == "Inception (ImageNet)":
        inception_classifier()
    elif page == "ResNet18 (Коты и Собаки)":
        cat_dog_classifier()
    else:
        skin_formation_classifier()

if __name__ == "__main__":
    main()
