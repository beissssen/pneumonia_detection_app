import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/chest_xray"  # <-- измени путь при необходимости
MODEL_PATH = "models/densenet121_pneumonia.pth"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4


def get_dataloaders(data_dir, batch_size=16):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # трансформации для увеличения данных
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes


def create_model(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # заменить классификатор
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    return model


def train_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    train_loader, val_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print("Classes:", classes)

    model = create_model(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_running_corrects.double() / val_total

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes
            }, MODEL_PATH)
            print(f"Best model updated, saved to {MODEL_PATH}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model()
