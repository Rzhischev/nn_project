import streamlit as st
from pages import page_inception, page_cat_dog, page_skin_formation

def main():
    st.sidebar.title("Выберите модель для классификации:")
    page = st.sidebar.radio("", ["Inception (ImageNet)", "ResNet18 (Коты и Собаки)", "Образования на коже"])

    if page == "Inception (ImageNet)":
        page_inception.show()
    elif page == "ResNet18 (Коты и Собаки)":
        page_cat_dog.show()
    else:
        page_skin_formation.show()

if __name__ == "__main__":
    main()



# ## main_app.py
# import streamlit as st
# import torchvision.transforms as transforms
# from PIL import Image
# import torch
# from torchvision import models

# import torch

# import torchvision
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms as T
# from torchvision import io
# import torchutils as tu
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# import streamlit as st
# from PIL import Image
# import requests
# from io import BytesIO

# #модели
# from models.model_test import model2
# from torchvision.models import inception_v3

# model = inception_v3(pretrained=True)
# model2.load_state_dict(torch.load('models/model-test-cat_dog.pt', map_location=torch.device('cpu')))

# # функции
# # Функция для загрузки изображения из URL
# def load_image_from_url(url):
#     response = requests.get(url)
#     img_data = BytesIO(response.content)
#     return Image.open(img_data)

# # Получение класса изображения
# def predict_image_class(url):
#     with torch.no_grad():
#         image = load_image_from_url(url)
#         outputs = model(image)
#         _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
#         class_name = labels[predicted_class.item()]
#         return class_name

# # image = load_image_from_url(url)
# # Получение класса изображения
# def predict_image_class(image):
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
#         class_name = labels[predicted_class.item()]
#         return class_name

# # Функция для первой страницы
# def inception_classifier():
#     st.write("Классификация произвольного изображения с помощью модели Inception")

#     uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])
#     url = st.text_input("Или введите URL изображения...")

#     image = None

#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
#     elif url:
#         try:
#             image = load_image_from_url(url)
#             st.image(image, caption='Loaded Image from URL.', use_column_width=True)
#         except:
#             st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

#     if image:  # Если изображение успешно загружено (будь то через файл или URL)
#         # Весь ваш код обработки изображения здесь
#         # Модель в бою
#         model.eval()

#         transform = transforms.Compose([
#             transforms.Resize((299, 299)),
#             transforms.ToTensor()
#         ])

#         # Загрузка меток классов ImageNet
#         LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
#         labels = requests.get(LABELS_URL).json()

#         image_tensor = transform(image).unsqueeze(0)
#         outputs = model(image_tensor)
#         _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
#         class_name = labels[predicted_class.item()]
#         st.write(f"Predicted class: {class_name}")
#     else:
#         st.write("Пожалуйста, загрузите изображение или предоставьте URL.")

# # Функция для второй страницы
# def cat_dog_classifier():
#     st.write("Классификация изображений котов и собак")

#     uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])
#     url = st.text_input("Или введите URL изображения...")

#     image = None

#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
#     elif url:
#         try:
#             image = load_image_from_url(url)
#             st.image(image, caption='Loaded Image from URL.', use_column_width=True)
#         except:
#             st.write("Ошибка при загрузке изображения по URL. Проверьте ссылку и попробуйте еще раз.")

#     if image:  # Если изображение успешно загружено
#         model2.eval()

#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])

#         image_tensor = transform(image).unsqueeze(0)
#         dict = {0:'cat', 1:'dog'}
#         output = model2(image_tensor).sigmoid().round()
#         st.write(f"Predicted class: {dict[output.item()]}")
#     else:
#         st.write("Пожалуйста, загрузите изображение или предоставьте URL.")  

# # Функция для третьей страницы
# def skin_formation_classifier():
#     st.write("Классификация образований# Функция для третьей страницы
# def skin_formation_classifier():
#     st.write("Классификация образований на коже")

#     # uploaded_file = st.file_uploader("Загрузите изображение кожи...", type=["jpg", "png", "jpeg"])

#     # if uploaded_file is not None:
#     #     image = Image.open(uploaded_file)
#     #     st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#     #     # Загрузите ваш код для классификации образований на коже на коже")

#     # uploaded_file = st.file_uploader("Загрузите изображение кожи...", type=["jpg", "png", "jpeg"])

#     # if uploaded_file is not None:
#     #     image = Image.open(uploaded_file)
#     #     st.image(image, caption='Uploaded Image.', use_column_width=True)
        
#     #     # Загрузите ваш код для классификации образований на коже

# # Основное приложение
# def main():
#     st.sidebar.title("Выберите модель для классификации:")
#     page = st.sidebar.radio("", ["Inception (ImageNet)", "ResNet18 (Коты и Собаки)", "Образования на коже"])

#     if page == "Inception (ImageNet)":
#         inception_classifier()
#     elif page == "ResNet18 (Коты и Собаки)":
#         cat_dog_classifier()
#     else:
#         skin_formation_classifier()

# if __name__ == "__main__":
#     main()
