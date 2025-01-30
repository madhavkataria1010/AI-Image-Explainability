import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import random
from tqdm import tqdm
import time
import argparse
import torch.nn.functional as F
import math

# ------------------------ Configuration ------------------------

batch_size = 64  # Adjust based on GPU memory
ignore_extensions = ['.json', '.txt', '.log']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    # Uncomment the following line if your model expects normalized inputs
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ------------------------ Dataset Definition ------------------------

class InferenceDataset(Dataset):
    """
    Dataset class for unlabeled images used for inference.
    """
    def __init__(self, root_dir, transform=None, ignore_extensions=None):
        self.root_dir = root_dir
        self.transform = transform
        self.ignore_extensions = ignore_extensions or ['.json', '.txt', '.log']
        self.images = self._load_images()

    def _load_images(self):
        """
        Load all image file paths from the root directory, ignoring specified extensions.
        """
        all_files = os.listdir(self.root_dir)
        image_files = [
            os.path.join(self.root_dir, file)
            for file in all_files
            if os.path.isfile(os.path.join(self.root_dir, file)) and
               not any(file.lower().endswith(ext) for ext in self.ignore_extensions)
        ]
        print(f'Found {len(image_files)} images for inference.')
        return image_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        filename = os.path.basename(img_path)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, filename

# ------------------------ Model Definition ------------------------

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

        grid: torch.Tensor = self.grid
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
        Compute coefficients of the curve that interpolates given points.
        Args:
            x (torch.Tensor): Shape (num_points, in_features)
            y (torch.Tensor): Shape (num_points, in_features, out_features)
        Returns:
            torch.Tensor: Coefficients (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, num_points, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, num_points, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

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
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)

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


from torchvision.models import efficientnet_b3

class MultiHeadEfficientNetB3(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.2):
        super(MultiHeadEfficientNetB3, self).__init__()
        # Load EfficientNet-B3 as the backbone
        # If using newer versions of torchvision, might need to specify weights explicitly, e.g.:
        # efficientnet = efficientnet_b3(weights='IMAGENET1K_V1') if pretrained else efficientnet_b3(weights=None)
        efficientnet = efficientnet_b3(pretrained=pretrained)
        self.backbone = efficientnet.features  
        self.flatten_dim = efficientnet.classifier[1].in_features
        self.fc = KANLinear(in_features=self.flatten_dim, out_features=128)
        
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
        class1_logits = self.fc2_class1(torch.relu(class_1_embed))
        
        class_2_embed = self.fc1_class2(x)
        class2_logits = self.fc2_class2(torch.relu(class_2_embed))
        
        return class1_logits, class2_logits


def classify_output(class1_prob, class2_prob, threshold=0.5):
    """Classify based on probabilities from both heads."""
    class1 = (class1_prob > threshold).float()
    class2 = (class2_prob > threshold).float()
    return torch.where(class1 + class2 == 0, 0, torch.where(class1 > 0, 1, 2))

criterion = torch.nn.BCEWithLogitsLoss()

# ------------------------ Inference Function ------------------------

def perform_inference(model, dataloader, device, output_json_path):
    """
    Perform inference on the dataset and save predictions to a JSON file.
    
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computation on.
        output_json_path (str): Path to save the JSON output.
    
    Returns:
        None
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Inference", unit="batch"):
            images = images.to(device)  # Move images to GPU
            class1_logit, class2_logit = model(images)
            
            # Apply sigmoid to convert logits to probabilities
            class1_prob = torch.sigmoid(class1_logit)
            class2_prob = torch.sigmoid(class2_logit)
            
            # Use classify_output to get predicted classes
            preds = classify_output(class1_prob, class2_prob)
            
            # Move tensors to CPU and convert to numpy arrays
            preds = preds.cpu().numpy().flatten()
            class1_prob = class1_prob.cpu().numpy().flatten()
            class2_prob = class2_prob.cpu().numpy().flatten()
            
            for filename, pred, c1p, c2p in zip(filenames, preds, class1_prob, class2_prob):
                label = "Real" if pred == 0 else "Fake"
                predictions.append({
                    'index': int(filename.rstrip(".png")),
                    'prediction': label,
                })
    
    # Save predictions to JSON
    predictions.sort(key=lambda x: x['index'])
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Predictions saved to {output_json_path}")

# ------------------------ Main Execution ------------------------

def main():
    parser = argparse.ArgumentParser(description="Process input paths.")
    parser.add_argument('-d','--data_dir',  default='/adobe/test_adobe/perturbed_images_32')
    parser.add_argument('-m','--model_path', default='/adobe/models/train.pth')
    parser.add_argument('-o', '--output', default='/adobe/results/73_task1.json', help='Output JSON file path (optional, default: predictions.json)')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path 
    output_json_path = args.output  

    inference_dataset = InferenceDataset(
        root_dir=data_dir,
        transform=data_transforms,
        ignore_extensions=ignore_extensions
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model = torch.compile(MultiHeadEfficientNetB3(pretrained=False)).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    perform_inference(model, inference_loader, device, output_json_path)

if __name__ == "__main__":
    main()
