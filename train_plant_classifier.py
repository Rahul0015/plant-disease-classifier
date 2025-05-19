import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy

# --- Constants ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 6
LEARNING_RATE = 0.0001
FINE_TUNE_EPOCHS_ADDITIONAL = 10
FINE_TUNE_LR = 0.00001

# --- Paths ---
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Augmentation and Preprocessing ---
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
}

# --- Load Data ---
image_datasets = {x: datasets.ImageFolder(os.path.join(base_dir, x), data_transforms[x]) for x in ['train', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4) for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

if NUM_CLASSES != len(class_names):
    raise ValueError(f"Mismatch in NUM_CLASSES ({NUM_CLASSES}) and found classes ({len(class_names)}: {class_names}). Check data folders.")

print(f"Classes found: {class_names}")
print(f"Training data size: {dataset_sizes['train']}")
print(f"Validation data size: {dataset_sizes['validation']}")

# --- Model Setup ---
def initialize_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model

# --- Training Function ---
def train_model(model, criterion, optimizer, num_epochs=EPOCHS, is_fine_tuning=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                filename = 'plant_disease_resnet50_finetuned_best.pth' if is_fine_tuning else 'plant_disease_resnet50_best_classifier.pth'
                # When using DataParallel, save the underlying module weights:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), filename)
                else:
                    torch.save(model.state_dict(), filename)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\nBest val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

# --- Plotting ---
def plot_history(history):
    acc, val_acc = history['train_acc'], history['val_acc']
    loss, val_loss = history['train_loss'], history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.savefig('training_validation_curves_pytorch.png')
    plt.show()

# --- Main ---
def main():
    model = initialize_model()

    # Wrap model for multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU or CPU for training.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.module.fc.parameters() if isinstance(model, nn.DataParallel) else model.fc.parameters(), lr=LEARNING_RATE)

    print("\n--- Initial Training ---")
    model, history1 = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

    # Save final model
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), 'plant_disease_resnet50_final_classifier.pth')
    else:
        torch.save(model.state_dict(), 'plant_disease_resnet50_final_classifier.pth')

    print("\n--- Fine-Tuning ---")
    # Unfreeze all parameters
    if isinstance(model, nn.DataParallel):
        for param in model.module.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer_ft = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

    model, history2 = train_model(model, criterion, optimizer_ft, num_epochs=FINE_TUNE_EPOCHS_ADDITIONAL, is_fine_tuning=True)

    # Save fine-tuned model
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), 'plant_disease_resnet50_final_finetuned.pth')
    else:
        torch.save(model.state_dict(), 'plant_disease_resnet50_final_finetuned.pth')

    # Combine histories and plot
    combined_history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'train_acc': history1['train_acc'] + history2['train_acc'],
        'val_loss': history1['val_loss'] + history2['val_loss'],
        'val_acc': history1['val_acc'] + history2['val_acc'],
    }
    plot_history(combined_history)

if __name__ == '__main__':
    main()
