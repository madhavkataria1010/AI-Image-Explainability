import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import io

def soft_threshold(image, threshold=0.2):
    """
    Apply soft thresholding to an image tensor.

    Parameters:
    - image (Tensor): Input image tensor.
    - threshold (float): The threshold value to apply during soft thresholding (default is 0.2).

    Returns:
    - Tensor: Thresholded image tensor.
    """
    # Apply soft thresholding: y = sign(x) * max(|x| - threshold, 0)
    thresholded_tensor = torch.sign(image) * torch.maximum(torch.abs(image) - threshold, torch.zeros_like(image))
    
    # Clamp values to be between 0 and 1
    return torch.clamp(thresholded_tensor, 0, 1) 

def webp_compress(image_tensor, quality=80):
    """
    Compress an image tensor into the WEBP format with the given quality.

    Parameters:
    - image_tensor (Tensor): Input image tensor.
    - quality (int): The quality of the compressed WEBP image (default is 80).

    Returns:
    - Tensor: Compressed image tensor in the same device as the input tensor.
    """
    image_pil = TF.to_pil_image(image_tensor.cpu()) 
    
    with io.BytesIO() as output:
        image_pil.save(output, format="WEBP", quality=quality)
        output.seek(0)
        
        compressed_image = Image.open(output)
        return TF.to_tensor(compressed_image).to(image_tensor.device)

def dkl_loss(logits_nat, logits_adv, weight, alpha, beta):
    """
    Compute the DKL loss between natural and adversarial logits.

    Parameters:
    - logits_nat (Tensor): The logits of the natural image.
    - logits_adv (Tensor): The logits of the adversarial image.
    - weight (Tensor): A weight tensor to scale the loss.
    - alpha (float): A scaling factor for the softmax cross-entropy (SCE) loss.
    - beta (float): A scaling factor for the mean square error (MSE) loss.

    Returns:
    - Tensor: The computed DKL loss.
    """
    num_classes = logits_nat.size(1)
    delta_n = logits_nat.view(-1, num_classes, 1) - logits_nat.view(-1, 1, num_classes)
    delta_a = logits_adv.view(-1, num_classes, 1) - logits_adv.view(-1, 1, num_classes)
    
    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * weight).sum() / logits_nat.size(0)
    loss_sce = -(F.softmax(logits_nat, dim=1).detach() * F.log_softmax(logits_adv, dim=-1)).sum(1).mean()
    
    return beta * loss_mse + alpha * loss_sce
