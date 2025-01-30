import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

# Configuration
num_epochs = 100
batch_size = 128
ignore_dir = []

classes = ['GAN-based', 'Nature', 'Diffusion-based']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = "/data/Adobe/training_data"
ai_folder = 'ai_60'
nature_folder = 'nature_60'
save_best_path = './models/multilabel_resnet50_best.pt'
save_path = './models/multilabel_resnet50_final.pt'
log_file = './logs/multilabel_resnet50.json'

patience = 10
best_val_loss = float('inf') 

# Data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None, split='train', ignore_dirs=None, ignore_extensions=None, sample_fraction=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.split = split
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else []
        self.ignore_extensions = ignore_extensions if ignore_extensions is not None else []
        self.sample_fraction = sample_fraction
        self.image_paths = []
        self.labels = []

        # Directories for different categories
        cifake_dirs = ['cifake']
        midjourney_dirs = ['midjourney']
        sdv5_dirs = ['sdv5']
        biggan_dirs = ['biggan']
        
        # Label mapping
        label_mapping = {
            "GAN-based": biggan_dirs,
            "Nature": cifake_dirs,
            "Diffusion-based": midjourney_dirs + sdv5_dirs + cifake_dirs
        }

        # Assign labels and collect file paths
        for label, dirs in label_mapping.items():
            for class_dir in dirs:
                for sub_dir in ['nature_60', 'ai_60']:
                    full_dir = os.path.join(dataset_path, class_dir, split, sub_dir)
                    print(f"Loading directory: {full_dir}")
                    if os.path.isdir(full_dir):
                        for file_name in os.listdir(full_dir):
                            file_path = os.path.join(full_dir, file_name)
                            if file_path.endswith(tuple(self.ignore_extensions)):
                                continue
                            if os.path.isfile(file_path):
                                # Assign labels based on directory structure
                                if sub_dir == 'nature_60':
                                    final_label = 1  # Nature
                                elif class_dir in biggan_dirs:
                                    final_label = 0  # GAN-based
                                else:
                                    final_label = 2  # Diffusion-based

                                self.image_paths.append(file_path)
                                self.labels.append(final_label)

        if self.sample_fraction:
            num_samples = int(len(self.image_paths) * self.sample_fraction)
            self.image_paths = self.image_paths[:num_samples]
            self.labels = self.labels[:num_samples]
        
        print(f"Label distribution: {self.split} - {set(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_data_loaders(dataset_path, batch_size=32, num_workers=0, ignore_dirs=None, ignore_extensions=None):
    train_dataset = ImageDataset(
        dataset_path,
        transform=data_transforms['train'],
        split='train',
        ignore_dirs=ignore_dir,
        ignore_extensions=ignore_extensions,
    )
    val_dataset = ImageDataset(
        dataset_path,
        transform=data_transforms['val'],
        split='val',
        ignore_dirs=ignore_dir,
        ignore_extensions=ignore_extensions,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader



################################################################################################
#######################################  Create Model ##########################################
################################################################################################

class ResNet50WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet50WithDropout, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Instantiate model
num_classes = len(classes)
dropout_prob = 0.3
model = ResNet50WithDropout(num_classes=num_classes, dropout_prob=dropout_prob)
model.to(device)


################################################################################################
#######################################  Training Phase ########################################
################################################################################################

model = torch.compile(model)

# Training loop
train_loader, val_loader = create_data_loaders(dataset_path, batch_size=batch_size)

optimizer = optim.AdamW(model.parameters(), lr=1.2e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=num_epochs, 
    steps_per_epoch=len(train_loader)
)

results = []
epochs_no_improve = 0
adversarial_prob = 0.3

results = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 30)

    # Training Phase
    model.train()
    train_loss, train_correct, total_samples, adv_accuracy = 0.0, 0, 0, 0
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast('cuda'):
                clean_logits = model(images)
                loss = F.cross_entropy(clean_logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, preds = clean_logits.max(1)
            train_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            pbar.set_postfix(train_loss=train_loss / total_samples, train_accuracy=train_correct / total_samples, adv_accuracy=adv_accuracy)

    train_accuracy = train_correct / total_samples
    train_loss /= total_samples
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast('cuda'):
                clean_logits = model(images)
                loss = F.cross_entropy(clean_logits, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = clean_logits.max(1)
            val_correct += torch.sum(preds == labels).item()
            val_samples += labels.size(0)

    val_accuracy = val_correct / val_samples
    val_loss /= val_samples
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

    epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    results.append(epoch_results)
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0 
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch + 1}")
    else:
        patience_counter += 1
        print(f"Patience Counter: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break