# Установка необходимых библиотек

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import requests
from io import BytesIO

# Загрузка предварительно обученной модели Inception
model = inception_v3(pretrained=True)
model.eval()  # Установка модели в режим оценки (не обучения)

# Функция для загрузки и обработки изображения
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(299),  # Размер, который требуется для Inception
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Нормализация для ImageNet
    ])
    return preprocess(image).unsqueeze(0)

# Загрузка меток классов ImageNet
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
labels = requests.get(LABELS_URL).json()

# Получение класса изображения
def predict_image_class(url):
    with torch.no_grad():
        image = load_image_from_url(url)
        outputs = model(image)
        _, predicted_class = outputs.max(1)  # Получаем класс с максимальной вероятностью
        class_name = labels[predicted_class.item()]
        return class_name

# Теперь вы можете предсказать класс изображения, используя:
url = "https://t1.gstatic.com/licensed-image?q=tbn:ANd9GcRRv9ICxXjK-LVFv-lKRId6gB45BFoNCLsZ4dk7bZpYGblPLPG-9aYss0Z0wt2PmWDb"
predicted_class_name = predict_image_class(url)
print(f"On the image: {predicted_class_name}")