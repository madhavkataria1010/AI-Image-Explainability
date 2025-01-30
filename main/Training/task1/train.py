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
import torchattacks
from tqdm import tqdm

# Configuration
num_epochs = 100
batch_size = 512
ignore_dir = []
data_dir = './training_data'
ai_folder = 'ai_og'
nature_folder = 'nature_og'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_val_accuracy = 0
print(device)
# Data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        # transforms.Resize((128, 128)),  # Ensure validation images are the same size
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
                "ai_og": 2 if source_dir in ['biggan', 'stylegan'] else 1,
                "nature_og": 0
            }

            for class_name, class_label in class_map.items():
                class_dir = os.path.join(split_path, class_name)
                print(class_dir)
                if not os.path.exists(class_dir):
                    continue

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
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
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


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )




import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define or import KANLinear and KAN classes here

class MultiHeadEfficientNetB3(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.2):
        super(MultiHeadEfficientNetB3, self).__init__()
        # Load EfficientNet-B3 as the backbone
        efficientnet = efficientnet_b3(pretrained=pretrained)
        self.backbone = efficientnet.features  # Extract feature extractor part
        self.flatten_dim = efficientnet.classifier[1].in_features  
        self.fc = KANLinear(in_features=self.flatten_dim,
            out_features=128)
        
        # Fully connected layers for Class 1 prediction
        self.fc1_class1 = nn.Linear(128, 64)
        self.dropout_class1 = nn.Dropout(p=dropout_p)
        self.fc2_class1 = nn.Linear(64, 1)
        self.drop_class1 = nn.Dropout(p=0.5)
        
        # Fully connected layers for Class 2 prediction
        self.fc1_class2 = nn.Linear(128, 64)
        self.dropout_class2 = nn.Dropout(p=dropout_p)
        self.fc2_class2 = nn.Linear(64, 1)
        self.drop_class2 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Shared feature extraction
        x = self.backbone(x)  
        x = x.view(-1, self.flatten_dim)
        
        x = self.fc(x)

        class_1_embed = self.fc1_class1(x)
        class_1_embed = self.dropout_class1(class_1_embed)
        class1_logits = self.fc2_class1(torch.relu(class_1_embed))
        # class1_logits = self.drop_class1(class1_logits)
        
        # Class 2 prediction
        class_2_embed = self.fc1_class2(x)
        class_2_embed = self.dropout_class2(class_2_embed)
        class2_logits = self.fc2_class2(torch.relu(class_2_embed))
        # class2_logits = self.drop_class2(class2_logits)
        
        return class1_logits, class2_logits



model = torch.compile(MultiHeadEfficientNetB3(pretrained=True)).to(device)

criterion = torch.nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scaler = torch.amp.GradScaler('cuda')

def classify_output(class1_prob, class2_prob, threshold=0.5):
    """Classify based on probabilities from both heads."""
    class1 = (class1_prob > threshold).float()
    class2 = (class2_prob > threshold).float()
    return torch.where(class1 + class2 == 0, 0, torch.where(class1 > 0, 1, 2))


log_file = "train.json"
results = []
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0


# Training and validation loop
for epoch in range(num_epochs):
   
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 30)
    
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

            # Track the training loss and accuracy
            train_loss += loss.item() * images.size(0)
            class1_prob = torch.sigmoid(class1_pred)
            class2_prob = torch.sigmoid(class2_pred)
            preds = classify_output(class1_prob, class2_prob)
            train_correct += torch.sum(preds.view(-1) == labels.view(-1)).item()
            total_samples += labels.size(0)
            

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
            class1_prob = torch.sigmoid(class1_pred)
            class2_prob = torch.sigmoid(class2_pred)
            preds = classify_output(class1_prob, class2_prob)
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
        torch.save(model.state_dict(), 'train.pth')
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
torch.save(model.state_dict(), 'train.pth')

