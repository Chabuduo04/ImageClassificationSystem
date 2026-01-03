import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(dataloader, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # Get number of classes from dataset if available
    try:
        num_classes = len(dataloader.dataset.classes)
    except AttributeError:
        # Fallback: get from model output shape (consumes one batch)
        with torch.no_grad():
            data_iter = iter(dataloader)
            sample_images, _ = next(data_iter)
            num_classes = model(sample_images[0:1].to(device)).shape[1]
    
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Calculate precision, recall and f1 for each class
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(0) - tp
    fn = confusion_matrix.sum(1) - tp
    
    precision = tp / (tp + fp + 1e-10)  # Add small epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Calculate macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }
