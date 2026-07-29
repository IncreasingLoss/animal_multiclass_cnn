"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path




def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


import torch
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics

def show_batch(data_loader, class_names=None, denormalize=False, figsize=(12, 8)):
    """
    Visualizes a random batch of images with optional ImageNet denormalization
    - denormalize: Whether to reverse ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    # Get batch and move to CPU
    images, labels = next(iter(data_loader))
    images = images.cpu()
    
    # Denormalize if requested
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)  # Maintain valid pixel range
    
    # Setup plot
    class_names = class_names or [str(i) for i in range(10)]
    batch_size = len(images)
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot images
    for ax, img, label in zip(axes, images, labels):
        # Convert tensor to numpy and permute dimensions
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(class_names[label])
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(batch_size, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

    
# training & testing loop
def train_step_CNN(MODEL: torch.nn.Module,
               LOADER: torch.utils.data.DataLoader,
               LOSSF: torch.nn.Module,
               ACCF: torchmetrics.Accuracy,
               OPTIM: torch.optim.Optimizer,
               SCHEDULER: None,
               DEVICE: torch.device):
    """Performs a CNN training step over a dataloader"""
    MODEL.train()
    train_loss, train_acc = 0, 0
    for image, label in LOADER:
        image, label = image.to(DEVICE), label.to(DEVICE)
        logits = MODEL(image)
        batch_loss = LOSSF(logits, label)
        train_loss += batch_loss
        train_acc += ACCF(logits.argmax(dim=1), label)
        OPTIM.zero_grad()
        batch_loss.backward()
        OPTIM.step()
        if SCHEDULER != None:
            SCHEDULER.step()
    train_acc /= len(LOADER)*0.01
    train_loss /= len(LOADER)
    return train_loss, train_acc


def test_step_CNN(MODEL: torch.nn.Module,
                  LOADER: torch.utils.data.DataLoader,
                  LOSSF: torch.nn.Module,
                  ACCF: torchmetrics.Accuracy,
                  DEVICE: torch.device):
    """Performs a CNN testing step over a dataloader"""
    MODEL.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for image, label in LOADER:
            image, label = image.to(DEVICE) , label.to(DEVICE)
            logits = MODEL(image)
            batch_loss = LOSSF(logits, label)
            test_loss += batch_loss
            test_acc += ACCF(logits.argmax(dim=1), label)
        test_acc /= len(LOADER)*0.01
        test_loss /= len(LOADER)
        return test_loss, test_acc
    



from tqdm.auto import tqdm
import torchmetrics
def model_training_CNN(MODEL: torch.nn.Module,
               LOADER_TRAIN: torch.utils.data.DataLoader,
               LOADER_TEST: torch.utils.data.DataLoader,
               LOSSF: torch.nn.Module,
               ACCF: torchmetrics.Accuracy,
               OPTIM: torch.optim.Optimizer,
               DEVICE: torch.device,
               SCHEDULER: None,
               EPOCHS_TRAIN: int,
               Epochs_RETURN: int):
    """Combines "train_step_CNN" & "test_step_CNN" into one training function"""
    
    evaluation_dictionary = {"train_loss": [],
                             "train_acc": [], 
                             "test_loss": [],
                             "test_acc": [],
                             "epoch":[]}

    for epoch in tqdm(range(EPOCHS_TRAIN)):

        train_loss, train_acc = train_step_CNN(MODEL, LOADER_TRAIN, LOSSF, ACCF, OPTIM, SCHEDULER, DEVICE)    
        test_loss, test_acc = test_step_CNN(MODEL, LOADER_TEST, LOSSF, ACCF, DEVICE)    
        
        #append to eval_dict
        if epoch%Epochs_RETURN == 0:
            evaluation_dictionary["train_loss"].append(train_loss.cpu().detach().numpy())
            evaluation_dictionary["train_acc"].append(train_acc.cpu())
            evaluation_dictionary["test_loss"].append(test_loss.cpu())
            evaluation_dictionary["test_acc"].append(test_acc.cpu())
            evaluation_dictionary["epoch"].append(epoch)
            print(f"Epoch: {epoch} || train_loss: {train_loss:.4f} || train_acc: {train_acc:.4f} || test_loss: {test_loss:.4f} || test_acc: {test_acc:.4f}")
    return evaluation_dictionary




from pathlib import Path
def get_classes_from_folder(folder:str):
    """Returns subfolders in a main folder/directory"""
    path = Path(f"{folder}")
    list_classes = []
    for label in path.iterdir():
        label = label.stem
        list_classes.append(label)
    return list_classes        




import os
from PIL import Image

def resize_images(directory, target_px, convert_to_jpg=False):
    """
    Resize images while preserving aspect ratio based on smallest dimension.
    Converts palette images to RGBA and optionally to RGB JPEGs.
    
    Args:
        directory (str): Root directory with images
        target_px (int): Target size for smallest dimension
        convert_to_jpg (bool): Convert to JPEG if True
    """
    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}

    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_ext:
                continue

            path = os.path.join(root, file)
            
            try:
                with Image.open(path) as img:
                    # Convert palette to RGBA first
                    if img.mode == 'P':
                        img = img.convert('RGBA')

                    # Handle JPEG conversion
                    if convert_to_jpg:
                        if img.mode in ('RGBA', 'LA'):
                            # Remove alpha channel
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[-1])
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')

                    # Calculate new dimensions
                    width, height = img.size
                    min_dim = min(width, height)
                    
                    if min_dim != target_px:  # Resize only if needed
                        ratio = target_px / min_dim
                        new_w = int(width * ratio)
                        new_h = int(height * ratio)
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # Save with appropriate format
                    if convert_to_jpg:
                        new_path = os.path.splitext(path)[0] + '.jpg'
                        img.save(new_path, 'JPEG', quality=85, optimize=True)
                    else:
                        img.save(path, img.format)

                    print(f"Resized {path} to {img.size}")

            except Exception as e:
                print(f"Error processing {path}: {str(e)}")



import os

def return_class_imbalance(main_dir: str) -> list:
    """
    Counts images in all immediate subdirectories of a main directory.
    
    Args:
        main_dir (str): Path to the main directory containing subdirectories with images
        
    Returns:
        list: Counts of images in each immediate subdirectory, sorted alphabetically
              by subdirectory name
    """
    number_list = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
    
    # Validate input directory
    if not os.path.isdir(main_dir):
        raise ValueError(f"Directory {main_dir} does not exist or is not a directory")
    
    # Get sorted list of immediate subdirectories
    subdirs = sorted([d for d in os.listdir(main_dir) 
                     if os.path.isdir(os.path.join(main_dir, d))])
    
    # Count images in each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        count = 0
        
        # Walk through all files in subdirectory tree
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    count += 1
                    
        number_list.append(count)
    
    return number_list
