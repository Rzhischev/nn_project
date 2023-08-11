import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Определение преобразований
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}




# 1. Загрузка и модификация модели

device='cpu'

def load_resnet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_cancer = models.resnet50(pretrained=False)
    num_features = resnet_cancer.fc.in_features
    resnet_cancer.fc = nn.Linear(num_features, 2)  # Замена последнего слоя для классификации котов и собак
    resnet_cancer.to(device)

    # 2. Загрузка сохраненных весов
    resnet_cancer.load_state_dict(torch.load('models/skin_cancer_resnet50_weights.pt', map_location=device)) #model-test-cat_dog
    resnet_cancer.eval()
    return resnet_cancer


# Функция для классификации изображений из URL
def predict_from_url(url, model):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    preprocess = data_transforms['test']
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return 'Benign' if predicted.item() == 0 else 'Malignant'
    
# # Тестирование
# url = "https://dermatologs.com/images/others/divkrasainas-dzimumzimes.jpg"
# prediction = predict_from_url(url, resnet_cancer)
# print(f"This image likely contains a: {prediction}")
