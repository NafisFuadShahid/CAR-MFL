import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=True):
        super(ImageEncoder, self).__init__()
        # Load ResNet50, remove final fully connected layer
        resnet = resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # remove last fc layer
        self.features = nn.Sequential(*modules)
        in_features = resnet.fc.in_features
        self.proj = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x
