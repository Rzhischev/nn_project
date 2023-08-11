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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, inception_v3
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

model.classifier[3] = nn.Linear(1024,1)

model2 = model
