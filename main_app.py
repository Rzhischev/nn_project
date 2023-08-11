# main_app.py
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision import models

# Функция для первой страницы
def inception_classifier():
    st.write("Классификация произвольного изображения с помощью модели Inception")

    uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Загрузка модели и классификация
        model = models.inception_v3(pretrained=True)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)
        output = model(image_tensor)
        _, prediction = torch.max(output, 1)
        
        # Используйте labels из ImageNet для вывода результатов
        st.write(f"Predicted class: {prediction.item()}")

# Функция для второй страницы
def cat_dog_classifier():
    st.write("Классификация изображений котов и собак")

    # Загрузите ваш код для дообученной ResNet18 и классификации

# Функция для третьей страницы
def skin_formation_classifier():
    st.write("Классификация образований на коже")

    uploaded_file = st.file_uploader("Загрузите изображение кожи...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Загрузите ваш код для классификации образований на коже

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
