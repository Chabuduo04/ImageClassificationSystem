import torch
import torch.nn as nn
import torchvision.models as models

# --------- 1. 基础 CNN ---------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# --------- 2. ResNet18 ---------
def get_resnet18(num_classes=10):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --------- 3. MobileNetV2 ---------
def get_mobilenet_v2(num_classes=10):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model
