import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import random
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from kan.kan import KANLinear, KAN

# Configuration
num_epochs = 100
batch_size = 128
ignore_dir = []
data_dir = '../training_data'
ai_folder = 'ai_60'
nature_folder = 'nature_60'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = './models/KAN1_b3_og.pth'
save_best_path = './models/KAN1_b3_best_og.pth'
torch.backends.cudnn.benchmark = True

best_val_accuracy = 0
print(device)
# Data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((32, 32)),  # Resize all images to 128x128
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((32  , 32)),  # Ensure validation images are the same size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', ignore_dirs=None, ignore_extensions=None, sample_fraction=None, seed=42):
        self.ignore_dirs = ignore_dirs or ['.ipynb_checkpoints', '__pycache__']
        self.ignore_extensions = ignore_extensions or ['.json', '.txt', '.log']
        self.images = []
        self.labels = []
        self.sources = []
        self.transform = transform
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.split = split 
        self._scan_directory(root_dir)

    def _scan_directory(self, root_dir):
        random.seed(self.seed)
        for source_dir in os.listdir(root_dir):
            if source_dir in self.ignore_dirs:
                continue

            source_path = os.path.join(root_dir, source_dir)
            if not os.path.isdir(source_path):
                continue

            split_path = os.path.join(source_path, self.split) 
            if not os.path.exists(split_path):
                continue
            
            # Class mapping
            class_map = {
                "ai_60": 2 if (source_dir in ['biggan', 'stylegan','misc'] )else 1,
                "nature_60": 0
            }

            print(class_map,source_dir)

            for class_name, class_label in class_map.items():
                class_dir = os.path.join(split_path, class_name)
                if not os.path.exists(class_dir):
                    continue
                print(class_dir)
                all_images = [
                    os.path.join(class_dir, img_name)
                    for img_name in os.listdir(class_dir)
                    if not any(img_name.endswith(ext) for ext in self.ignore_extensions)
                ]

                if self.sample_fraction:
                    num_samples = max(1, int(len(all_images) * self.sample_fraction))
                    sampled_images = random.sample(all_images, num_samples)
                else:
                    sampled_images = all_images

                for img_path in sampled_images:
                    try:
                        self.images.append(img_path)
                        self.labels.append(class_label)
                        self.sources.append(source_dir)
                    except (IOError, SyntaxError) as e:
                        print(f"Skipping invalid image {img_path}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.to(device), self.labels[idx], self.sources[idx]

# Create DataLoaders
def create_data_loaders(root_dir, batch_size=32, num_workers=None, ignore_dirs=None, ignore_extensions=None):
    if num_workers is None:
        num_workers = 0
    
    # Create training dataset
    train_dataset = ImageDataset(
        root_dir, 
        transform=data_transforms['train'],
        split='train',  
        ignore_dirs=ignore_dirs,
        ignore_extensions=ignore_extensions
    )
    
    # Create validation dataset
    val_dataset = ImageDataset(
        root_dir, 
        transform=data_transforms['val'],
        split='val', 
        ignore_dirs=ignore_dirs,
        ignore_extensions=ignore_extensions
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        # pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        # pin_memory=True
    )
    
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(data_dir, batch_size=batch_size)
print(".................................................................................................")
print(batch_size, len(train_loader), len(val_loader))
print(".................................................................................................")

from torchvision.models import efficientnet_b3
import torch
import torch.nn.functional as F
import math




import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadEfficientNetB3(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.2):
        super(MultiHeadEfficientNetB3, self).__init__()
        # Load EfficientNet-B3 as the backbone
        efficientnet = efficientnet_b3(pretrained=pretrained)
        self.backbone = efficientnet.features  # Extract feature extractor part
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_dim = efficientnet.classifier[1].in_features  # Get the input size for the classifier
        # self.fc = KANLinear(in_features=self.flatten_dim, out_features=128)
        self.fc = nn.Linear(self.flatten_dim, 128)
        
        # Fully connected layers for Class 1 prediction
        self.fc1_class1 = nn.Linear(128, 64)
        self.dropout_class1 = nn.Dropout(p=dropout_p)
        self.fc2_class1 = nn.Linear(64, 1)
        
        # Fully connected layers for Class 2 prediction
        self.fc1_class2 = nn.Linear(128, 64)
        self.dropout_class2 = nn.Dropout(p=dropout_p)
        self.fc2_class2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Shared feature extraction
        x = self.backbone(x)  
        x = self.global_pool(x) 
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        # Class 1 prediction with Dropout
        class1_hidden = torch.relu(self.fc1_class1(x))
        class1_hidden = self.dropout_class1(class1_hidden)
        class1_pred = self.fc2_class1(class1_hidden)
        
        # Class 2 prediction with Dropout
        class2_hidden = torch.relu(self.fc1_class2(x))
        class2_hidden = self.dropout_class2(class2_hidden)
        class2_pred = self.fc2_class2(class2_hidden)
        
        return class1_pred, class2_pred




def classify_output(class1_prob, class2_prob, threshold=0.5):
    """Classify based on probabilities from both heads."""
    class1 = (class1_prob > threshold).float()
    class2 = (class2_prob > threshold).float()
    return torch.where(class1 + class2 == 0, 0, torch.where(class1 > 0, 1, 2))


# Model setup (Using ResNet-18)
# Model setup (Using the multi-head ResNet-18 defined earlier)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiHeadEfficientNetB3(pretrained=True).to(device)

criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
# optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=0.1, weight_decay=0.0005,nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=num_epochs, 
    steps_per_epoch=len(train_loader)
)

scaler = torch.amp.GradScaler('cuda')

# fgsm_attack = torchattacks.FGSM(model, eps=0.007)
# pgd_attack = torchattacks.PGD(model, eps=0.03, alpha=0.01, steps=40)

log_file = "./KAN1_b3_adver_drop_linear_sgd1.json"
results = []
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
adversarial_prob = 0.5

model = torch.compile(
        model
    )

# Training and validation loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 30)
    
    # Training phase with adversarial attack
    model.train()
    train_loss, train_correct = 0.0, 0
    total_samples = 0
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Forward pass and loss computation
            with torch.amp.autocast('cuda'):
                class1_pred, class2_pred = model(images)
                labels_class1 = (labels == 1).float().unsqueeze(1)
                labels_class2 = (labels == 2).float().unsqueeze(1)
                
                # Compute BCE loss for each head
                loss_class1 = criterion(class1_pred, labels_class1)
                loss_class2 = criterion(class2_pred, labels_class2)
                
                # Total loss
                loss = loss_class1 + loss_class2
            
            # Backward pass and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Track the training loss and accuracy
            train_loss += loss.item() * images.size(0)
            preds = classify_output(class1_pred, class2_pred)
            train_correct += torch.sum(preds.view(-1) == labels.view(-1)).item()
            total_samples += labels.size(0)
            
            # print(class1_pred.size())
            # print(preds.shape)
            # print(labels.shape)
            # print(torch.sum(preds == labels))
            # print(train_correct)
            # print(total_samples)
            pbar.set_postfix(train_loss=train_loss / total_samples, train_accuracy=train_correct / total_samples)
    # Calculate average training accuracy and loss
    train_accuracy = train_correct / total_samples
    train_loss /= total_samples
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    
    # Validation phase
    model.eval()
    val_loss, val_correct = 0.0, 0
    total_samples = 0
    with tqdm(val_loader, desc="Validation", unit="batch") as pbar:
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                class1_pred, class2_pred = model(images)
                labels_class1 = (labels == 1).float().unsqueeze(1)
                labels_class2 = (labels == 2).float().unsqueeze(1)
                
                # Compute BCE loss for each head
                loss_class1 = criterion(class1_pred, labels_class1)
                loss_class2 = criterion(class2_pred, labels_class2)
                
                # Total validation loss
                loss = loss_class1 + loss_class2
            
            val_loss += loss.item() * images.size(0)
            preds = classify_output(class1_pred, class2_pred)
            val_correct += torch.sum(preds.view(-1) == labels.view(-1)).item()
            total_samples += labels.size(0)
            
            pbar.set_postfix(val_loss=val_loss / total_samples, val_accuracy=val_correct / total_samples)

    val_accuracy = val_correct / total_samples
    val_loss /= total_samples
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
    
    # Scheduler step
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), save_best_path)
        print("Best Model saved")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Log results
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

# Save final model
torch.save(model.state_dict(), save_path)
