import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torchvision.transforms as T
from torchattacks import PGD, FGSM
from torchvision.models import resnet50
import random
import albumentations as A
import numpy as np
from torchvision import models, transforms, datasets
from PIL import Image
import torch.nn as nn

class ImageProcessor:
    def __init__(
        self, 
        input_folder, 
        output_folder, 
        min_object_coverage=30, 
        max_objects=2, 
        model_path='yolov10n.pt'
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_object_coverage = min_object_coverage
        self.max_objects = max_objects
        
        os.makedirs(output_folder, exist_ok=True)
        
        self.model = YOLO(model_path)
        self.model.fuse() 
        print(input_folder, output_folder)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attack_model = resnet50(pretrained=True)

        self.attack_model = self.attack_model.to(self.device).eval()

        self.attack_epsilon = 6/255
        self.attack_methods = [
            FGSM(self.attack_model, eps=self.attack_epsilon),
            PGD(self.attack_model, eps=self.attack_epsilon, alpha=2/255, steps=10)
        ]
        
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.JPEG', '.PNG')

        self.transform = A.Compose([
            # Geometric transformations
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1)),
            A.HorizontalFlip(p=0.5),
            
            #Color space transformations
            A.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.1, 
                p=0.4
            ),
            
            # Advanced color transformations
            A.OneOf([
                A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=8, 
                    sat_shift_limit=8, 
                    val_shift_limit=8, 
                    p=0.3
                )
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(var_limit=(2, 4.0), p=0.2),
            ], p=0.2)
        ])

    def analyze_image_objects(self, results):
        filtered_objects = [
            box for box in results[0].boxes 
            if box.conf > 0.5  # High confidence objects only
        ]
        
        return filtered_objects

    def calculate_object_coverage(self, image, objects):
        height, width = image.shape[:2]
        total_image_area = height * width

        detected_object_area = sum(
            (box.xyxy[0][2].cpu() - box.xyxy[0][0].cpu()).item() * 
            (box.xyxy[0][3].cpu() - box.xyxy[0][1].cpu()).item() 
            for box in objects
        )

        coverage_percentage = (detected_object_area / total_image_area) * 100
        return coverage_percentage
    
    def apply_adversarial_attack(self, image):
        # Check if the image is a numpy ndarray
        

        if random.random() < 0.2:
            transform = A.Compose([
            A.CLAHE(p=0.5),
            A.Sharpen(p=0.1),  
            ])
            transformed = self.transform(image=image)
            image = transformed['image']
            return image

        if isinstance(image, np.ndarray):
            # Convert the ndarray to a PIL image
            image = Image.fromarray(image)

        if self.attack_model is None:
            print("Error: Attack model is not properly initialized.")
            return None

        # Transformation pipeline
        


        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Check if the tensor is valid
        if image_tensor is None or image_tensor.size(0) != 1:
            print("Error: Image transformation failed, invalid tensor.")
            return None
    
        attack = random.choice(self.attack_methods)
            
        self.attack_methods = ['FGSM', 'PGD','Both']
        probabilities = [0.33,0.33, 0.33]
        attack = random.choices(self.attack_methods, probabilities, k=1)[0]

        with torch.no_grad():
            output = self.attack_model(image_tensor) 
            _, predicted_label = torch.max(output, 1)
        try:
            if attack == 'FGSM':
                attack_method = FGSM(self.attack_model, eps=self.attack_epsilon)
                adversarial_image = attack_method(image_tensor, predicted_label)
            elif attack == 'PGD':
                attack_method = PGD(self.attack_model, eps=self.attack_epsilon, alpha=2/255, steps=10)
                adversarial_image = attack_method(image_tensor,predicted_label)
            elif attack == 'Both':
                fgsm_attack = FGSM(self.attack_model, eps=self.attack_epsilon)
                adversarial_image_fgsm = fgsm_attack(image_tensor,predicted_label)
        
                pgd_attack = PGD(self.attack_model, eps=self.attack_epsilon, alpha=2/255, steps=5)
                adversarial_image_pgd = pgd_attack(image_tensor,predicted_label)
    
                adversarial_image = (adversarial_image_fgsm + adversarial_image_pgd) / 2
        except Exception as e:
            print(f"Error during attack application: {str(e)}")
            return None

        # Convert adversarial image back to numpy format and ensure it is in the correct range
        adversarial_image = adversarial_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        # print(adversarial_image.min(), adversarial_image.max())
        adversarial_image = (adversarial_image - adversarial_image.min()) / (adversarial_image.max() - adversarial_image.min())
        # print(adversarial_image.min(), adversarial_image.max())
        adversarial_image = (adversarial_image * 255).astype(np.uint8)
        return adversarial_image



    def process_image(self, filename):
        input_path = os.path.join(self.input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(self.output_folder, output_filename)

        try:
            # Robust image reading
            img = cv2.imread(input_path)
            if img is None:
                print(f"[ERROR] Could not read image: {filename}")
                return None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Efficient object detection
            results = self.model(img_rgb, verbose=False)
            detected_objects = self.analyze_image_objects(results)
            
            # Filtering conditions
            if (len(detected_objects) > self.max_objects or 
                len(detected_objects) == 0):
                return None

            object_coverage = self.calculate_object_coverage(img_rgb, detected_objects)
            
            if object_coverage < self.min_object_coverage:
                return None

            # Image preprocessing
            # cropped imaged to 1:1
            height, width = img_rgb.shape[:2]
            new_edge = min(width, height)
            left = (width - new_edge) // 2
            top = (height - new_edge) // 2
            img_cropped = img_rgb[top:top+new_edge, left:left+new_edge]

            # transformed = self.transform(image=img_cropped)
            # img_cropped = transformed['image']
            
            
            if random.random() < 0.55:
                img_cropped = self.apply_adversarial_attack(img_cropped)
        

            img_resized = cv2.resize(img_cropped, (32, 32), interpolation=cv2.INTER_LINEAR)
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            return {
                'filename': filename,
                'object_count': len(detected_objects),
                'object_coverage': object_coverage,
                'object_classes': [results[0].names[int(obj.cls)] for obj in detected_objects]
            }

        except Exception as e:
            print(f"[ERROR] Processing {filename}: {e}")
            return None

    def process_images(self, max_workers=None, max_images=None):
        max_workers = max_workers or os.cpu_count()
        
        # Identify supported image files and shuffle for randomness
        files_to_process = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(self.supported_formats)
        ]
        random.shuffle(files_to_process)

        processed_images = []
        total_detected_objects = 0

        # Intelligent progress tracking
        with tqdm(total=len(files_to_process), 
                desc="Processing Images", 
                unit="image",
                dynamic_ncols=True) as progress_bar:
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_image, filename): filename 
                    for filename in files_to_process
                }

                for future in as_completed(futures):
                    if (max_images != None) and (len(processed_images) >= max_images):
                        # Cancel remaining futures if object limit reached
                        for remaining_future in futures:
                            remaining_future.cancel()
                        break

                    result = future.result()
                    
                    if result:
                        processed_images.append(result)
                        total_detected_objects += result['object_count']
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Objects': total_detected_objects,
                        'Processed': len(processed_images)
                    })

                    # Break condition check after processing
                    if len(processed_images) >= max_images:
                        break

        # Comprehensive logging
        print(f"\n[SUMMARY]")
        print(f"Total images processed: {len(processed_images)}")
        print(f"Total detected objects: {total_detected_objects}")
        
        if processed_images:
            object_counts = [img['object_count'] for img in processed_images]
            print(f"Object count distribution: {np.unique(object_counts, return_counts=True)}")
            print(f"Average object coverage: {np.mean([img['object_coverage'] for img in processed_images]):.2f}%")

        return processed_images


# Execution
processor = ImageProcessor(
    input_folder="./midjourney/train/ai",
    output_folder="./midjourney/train/ai_60",
    min_object_coverage=7,
    max_objects=15
)

processed_images = processor.process_images(max_workers=None,max_images=50000)
