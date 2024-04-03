# Imports

import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image

import matplotlib.pyplot as plt

from PGD import MyPGD

# Data
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
print(device)

# Load the data

LABEL_LIST = tuple(open('./data/imagenet_labels.txt', 'r').read().split('\n'))
LABEL_LIST = [x.replace('{',"").replace('\'',"").replace(',',"").replace('-'," ").replace('_'," ") for x in LABEL_LIST]

model = models.mobilenet_v3_small(pretrained=True).eval().to(device)


image = Image.open('./data/kobe.jpg')
x = torchvision.transforms.functional.to_tensor(image)
x = transforms.Resize(size=(256, 256))(x)
x = x.unsqueeze(0).to(device)
PGD_ = MyPGD(model)
output = model(x)
l = nn.Softmax(dim=1)(output).max(1)[1]

eps = 0.02
x_PGD = PGD_(x, l, eps=eps)

idx = 0
fig = plt.figure(figsize=(11,7), dpi=300)

plt.subplot(1,3,1)
plt.imshow(x[idx].squeeze().detach().cpu().permute(1,2,0))
plt.axis('off')
plt.title(f'Original\nLabel {LABEL_LIST[nn.Softmax(dim=1)(model(x)).max(1)[1].item()]}')

plt.subplot(1,3,2)
plt.imshow(15 * (x_PGD[idx] - x[idx]).squeeze().detach().cpu().permute(1,2,0))
plt.axis('off')
plt.title(f'Perturbation\nEpsilon {eps}')

plt.subplot(1,3,3)
plt.imshow(x_PGD[idx].squeeze().detach().cpu().permute(1,2,0))
plt.axis('off')
plt.title(f'PGD\nPrediction {LABEL_LIST[nn.Softmax(dim=1)(model(x_PGD)).max(1)[1].item()]}')

plt.show()
