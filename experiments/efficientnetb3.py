import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms, datasets
from torch.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import json
import torch.nn.functional as F
from PIL import Image
from timm import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ",device)

dataset_path = "/data/Adobe/training_data"
ai_folder = 'ai_og'
nature_folder = 'nature_og'
ignore_dir = []
batch_size = 64
num_epochs = 100
save_best_path = './models/efficientnet_b3_best.pt'
save_path = './models/efficientnet_b3_final.pt'
log_file = './logs/efficientnet_b3.log'
patience = 10


"""
    Defined following transformation defined on respective images, training and validation images.
    used standard imagenet standard deviation and mean
"""
transform_config = {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }

class ImageDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        transform=None, 
        ignore_dirs=None,
        ignore_extensions=None,
        sample_fraction=None, 
        seed=42   
    ):

        self.ignore_dirs = ignore_dirs or [
            '.ipynb_checkpoints', 
            '__pycache__'
        ]
        
        self.ignore_extensions = ignore_extensions or [
            '.json', 
            '.txt', 
            '.log', 
        ]
        
        self.images = []
        self.labels = []
        self.sources = []
        self.transform = transform
        self.sample_fraction = sample_fraction
        self.seed = seed
        
        self._scan_directory(root_dir)
    
    def _scan_directory(self, root_dir):
        random.seed(self.seed)

        for source_dir in os.listdir(root_dir):

            if source_dir in self.ignore_dirs:
                continue
            
            source_path = os.path.join(root_dir, source_dir)
            
            if not os.path.isdir(source_path):
                continue
            
            for split in ['train', 'val']:
                split_path = os.path.join(source_path, split)
                print(split_path,sep='\n')
                if not os.path.exists(split_path):
                    continue
                for label, class_name in enumerate([ai_folder, nature_folder]):            
                    class_dir = os.path.join(split_path, class_name)
                    if not os.path.exists(class_dir):
                        continue
                    
                    all_images = [
                        os.path.join(class_dir, img_name)
                        for img_name in os.listdir(class_dir)
                        if not any(img_name.endswith(ext) for ext in self.ignore_extensions)
                    ]
                    
                    # Apply sampling if sample_fraction is provided
                    if self.sample_fraction:
                        num_samples = max(1, int(len(all_images) * self.sample_fraction))  # At least 1 image
                        sampled_images = random.sample(all_images, num_samples)
                    else:
                        sampled_images = all_images
                    
                    # Validate and add sampled images
                    for img_path in sampled_images:
                        try:
                            self.images.append(img_path)
                            self.labels.append(label)
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
        
        return image, self.labels[idx], img_path
    
def create_data_loaders(
    root_dir, 
    batch_size=32, 
    num_workers=None,
    ignore_dirs=None,
    ignore_extensions=None
):
    if num_workers is None:
        num_workers = (torch.cuda.device_count() * 8) if torch.cuda.is_available() else 4

    
    # Collect train and test datasets
    train_datasets, test_datasets = [], []
    
    for split, transform_type in [('train', 'train'), ('test', 'val')]:
        multi_source_dataset = ImageDataset(
            root_dir, 
            transform=transform_config[transform_type],
            ignore_dirs=ignore_dir,
            ignore_extensions=None,
            sample_fraction = None,
        )
        
        if split == 'train':
            train_datasets.append(multi_source_dataset)
        else:
            test_datasets.append(multi_source_dataset)
    
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        combined_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        combined_test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(dataset_path,batch_size=batch_size)

print("Train and Validation Data Loaders created")
print("Train Data Loader Length: ",len(train_loader))
print("Validation Data Loader Length: ",len(val_loader))

################################################################################################
#######################################  Create Model ##########################################
################################################################################################

"""
    Fine-tuned Efficient Net b3 model on our training dataset 
"""
model = create_model('efficientnet_b3', pretrained=True, num_classes=2) 
model.conv_stem = nn.Conv2d(3, model.conv_stem.out_channels, kernel_size=3, stride=1, padding=1)
model = model.to(device)


################################################################################################
#######################################  Training Phase ########################################
################################################################################################

optimizer = optim.AdamW(model.parameters(), lr=1.2e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing = 0.2)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=num_epochs, 
    steps_per_epoch=len(train_loader)
)


total_params = sum(p.numel() for p in model.parameters())
param_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024**2)
    
best_val_loss = np.inf
patience_counter = 0
adversarial_prob = 0.5

results = []

print("Starting training")
for epoch in range(num_epochs):
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    train_samples = 0

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        with autocast('cuda'):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics calculation
        _, predicted = logits.max(1)
        accuracy = (predicted == labels).float().mean().item()
        train_loss += loss.item() * labels.size(0)
        train_accuracy += accuracy * labels.size(0)
        train_samples += labels.size(0)

        # Logging batch-level metrics
        if batch_idx % 100 == 0:
            batch_type = "Both"
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Accuracy: {accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Batch Type: {batch_type}')


    avg_train_loss = train_loss / train_samples
    avg_train_accuracy = train_accuracy / train_samples

    # Validation Phase
    model.eval()
    val_loss, val_accuracy = 0.0, 0.0
    val_samples = 0
    all_preds, all_labels = [], []

    
    for images, labels, _ in val_loader:
        images, labels = images.to(device), labels.to(device)

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        accuracy = (predicted == labels).float().mean().item()
        val_loss += loss.item() * labels.size(0)
        val_accuracy += accuracy * labels.size(0)
        val_samples += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Validation metrics
    avg_val_loss = val_loss / val_samples
    avg_val_accuracy = val_accuracy / val_samples
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"\nEpoch {epoch} Summary:")
    print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")


    epoch_results = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    results.append(epoch_results)
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Early stopping mechanism
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_best_path)
        print(f"Validation loss improved from {best_val_loss:.4f}. Model saved.")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve from {best_val_loss:.4f} \nPatience: {patience_counter}/{patience}")
        

    if patience_counter >= patience:
        print(f"Early stopping activated after epoch {epoch}")
        break

torch.save(model.state_dict(), save_path)