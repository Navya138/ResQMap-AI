import torch.nn as nn
import torchvision


def make_damage_model(pretrained=True, n_classes=4):
    """ResNet18 that takes the 6-channel (pre+post) building crop."""
    m = None
    if pretrained:
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        except Exception:
            print("(could not download pretrained weights - training from scratch)")
    if m is None:
        m = torchvision.models.resnet18(weights=None)
    old = m.conv1
    new = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
    new.weight.data = old.weight.data.repeat(1, 2, 1, 1) / 2
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m
