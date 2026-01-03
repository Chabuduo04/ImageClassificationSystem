import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models import SimpleCNN, get_resnet18, get_mobilenet_v2
from utils import train_one_epoch, evaluate

# ---------- 参数 ----------
MODEL_NAME = "cnn"   # cnn | resnet | mobilenet
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
NUM_CLASSES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 数据 ----------
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ---------- 模型 ----------
if MODEL_NAME == "cnn":
    model = SimpleCNN(NUM_CLASSES)
elif MODEL_NAME == "resnet":
    model = get_resnet18(NUM_CLASSES)
elif MODEL_NAME == "mobilenet":
    model = get_mobilenet_v2(NUM_CLASSES)
else:
    raise ValueError("Unknown model")

model.to(DEVICE)

# ---------- 训练配置 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- 训练 ----------
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )
    val_metrics = evaluate(
        model, val_loader, criterion, DEVICE
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} "
        f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}"
    )

# ---------- 保存模型 ----------
torch.save(model.state_dict(), f"checkpoints/{MODEL_NAME}.pth")
print("Model saved.")