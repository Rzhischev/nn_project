# nn_project
Elbrus Bootcamp | Phase-2 | Team project


## 🦸‍♂️Команда
- [Vladimir Kadnikov](https://github.com/vkadnikov92)
- [Grisha Rzhishchev](https://github.com/Rzhischev)

## 🎯 Задача: 

Разработайте multipage-приложение с использованием streamlit:

1. Классификация произвольного изображения с помощью модели Inception, обученной на датасете ImageNet. Самостоятельно не обучали, использоалпсь готовая модель из torchvision.models

2. Классификация изображений котов и собак при помощи дообученной модели ResNet18. Была подгружена предобученная часть из torchvision.models и заменён последний слой.

|Stage|Loss|Accuracy|
|---|---|---|
|train:| 0.2146| 0.9155|
|val:| 0.0854| 0.9770|
|train:| 0.2035| 0.9120|
|val:| 0.0815| 0.9790|
|train:| 0.1815| 0.9175|
|val:| 0.0686| 0.9770|
|train:| 0.1766| 0.9315|
|val:| 0.0624| 0.9790|
|train:| 0.1710| 0.9260|
|val:| 0.0605| 0.9810|

3. Классификация образований на коже:
    - Модель ResNet50
    - Датасет Skin Cancer: Malignant vs Benign
    - Количество эпох 7
