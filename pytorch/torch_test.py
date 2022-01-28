import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


seq_modules = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28*28, out_features=20),
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
print(input_image)
logits = seq_modules(input_image)
print(logits)