# Import standard libraries for file operations, regex, timing, and deep copying objects
import os
import re
import time
import copy

# Import utilities for path handling, data structures, visualization, and image processing
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from PIL import Image
# Import metrics for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import PyTorch and related libraries for deep learning
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import visualization and pre-trained model libraries
import seaborn as sns
import torchvision
from torchvision import models, datasets, transforms

# Import custom online augmentation function to enhance the dataset during training
from augmentation_libraries.online_augmentation import augment_image_without_seed

# Print versions and CUDA availability for debugging
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print(torch.cuda.is_available())

# Set up directory paths for the project
CURRENT_DIR = os.getcwd()
MAIN_FOLDER = Path(CURRENT_DIR).parent
OUTPUT_FOLDER = os.path.join(MAIN_FOLDER, 'aligned')  # Directory with aligned face images
FOLD_DATA = os.path.join(MAIN_FOLDER, 'fold_data')    # Directory with cross-validation fold information

# Set batch size for training and evaluation
BATCH_SIZE = 64

# Set up CUDA device if available for GPU acceleration
cuda_avail = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda_avail else "cpu")

# Print directory information for debugging
print(
    f"Current Directory: {CURRENT_DIR}\n",
    f"Main Folder: {MAIN_FOLDER}\n",
    f"Output Folder: {OUTPUT_FOLDER}\n",
    f"Fold Data Folder: {FOLD_DATA}\n",
)

def get_data_transforms():
    """
    Define image transformations for train, validation and test datasets.
    All images are resized to 224x224 and normalized using the same parameters.
    
    Returns:
        dict: Dictionary of transform compositions for 'train', 'val', and 'test'
    """
    normalize = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # Mean and std for normalization

    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ]),
    }

# Create data transformations
data_transforms = get_data_transforms()

class BasicImageDataset(Dataset):
    """
    Custom dataset class that supports online augmentation.
    Can generate multiple augmented versions of each image during training.
    """
    def __init__(self, image_paths, labels, transform=None, augment=False, num_augmentations=2):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels.
            transform (callable, optional): Transformations to apply to the images.
            augment (bool): Whether to apply online augmentation.
            num_augmentations (int): Number of augmented versions to create per image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.num_augmentations = num_augmentations
        
    def __len__(self):
        # Return the actual number of samples after augmentation
        if self.augment:
            return len(self.image_paths) * (self.num_augmentations)
        return len(self.image_paths)
        
    def get_original_len(self):
        """Return the number of original images (without augmentation)"""
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.
        If augmentation is enabled, will return an augmented version of the image.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (image, label) where image is a tensor and label is a tensor
        """
        # Calculate which original image to use
        if self.augment:
            original_idx = idx // self.num_augmentations
        else:
            original_idx = idx
            
        # Load the image and convert to RGB
        image = Image.open(self.image_paths[original_idx]).convert('RGB')
        label = torch.tensor(self.labels[original_idx], dtype=torch.long)

        if self.augment:
            # When augmenting, first apply augmentation (which already returns a tensor)
            augmented_tensor = augment_image_without_seed(image, final_resolution=(224, 224))
            
            # Skip ToTensor and only apply normalization if needed
            if self.transform:
                # Extract the normalization from transform and apply it directly
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        augmented_tensor = t(augmented_tensor)
                        
            return augmented_tensor, label

        # For non-augmented images, apply the full transform
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

def load_folds_dataset(image_root, fold_dir, fold_files):
    """
    Load image paths and labels from fold files for cross-validation.
    
    Args:
        image_root (str): Root directory containing user folders with images
        fold_dir (str): Directory containing fold files
        fold_files (list): List of fold file names to process
        
    Returns:
        tuple: (image_paths, labels) - lists of image paths and corresponding labels
    """
    image_paths = []
    labels = []

    for fold_file in fold_files:
        print(f"Reading fold file: {fold_file}")
        with open(os.path.join(fold_dir, fold_file), 'r') as f:
            next(f)  # Skip header line
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                user_id = parts[0]
                original_img_name = parts[1]
                gender = parts[4].lower()

                # Skip entries with invalid gender labels
                if gender not in ["m", "f"]:
                    continue
                # Convert gender to binary label: 0 for male, 1 for female
                label = 0 if gender == "m" else 1

                # Find the user's folder and locate the specific image
                user_folder = os.path.join(image_root, user_id)
                if not os.path.isdir(user_folder):
                    continue

                for file in os.listdir(user_folder):
                    if original_img_name in file:
                        full_path = os.path.join(user_folder, file)
                        if os.path.isfile(full_path):
                            image_paths.append(full_path)
                            labels.append(label)
                        break

    return image_paths, labels


def get_dataloaders(batch_size, train_folds, val_fold):
    """
    Create DataLoaders for training and validation datasets.
    
    Args:
        batch_size (int): Batch size for DataLoaders
        train_folds (list): List of fold file names for training
        val_fold (str): Fold file name for validation
        
    Returns:
        dict: Dictionary with 'train' and 'val' DataLoaders, or None if datasets are empty
    """
    # Load datasets for training and validation
    train_image_paths, train_labels = load_folds_dataset(OUTPUT_FOLDER, FOLD_DATA, train_folds)
    val_image_paths, val_labels = load_folds_dataset(OUTPUT_FOLDER, FOLD_DATA, [val_fold])

    # Create dataset objects - enable augmentation only for training
    train_dataset = BasicImageDataset(train_image_paths, train_labels, transform=data_transforms['train'], augment=True)
    val_dataset = BasicImageDataset(val_image_paths, val_labels, transform=data_transforms['val'], augment=False)

    print(f"Train size: {len(train_dataset)} (from {train_dataset.get_original_len()} original images)")
    print(f"Val size: {len(val_dataset)}")

    # Return None if either dataset is empty
    if train_dataset.get_original_len() == 0 or len(val_dataset) == 0:
        return None

    # Optimize DataLoader performance based on hardware availability
    num_workers = 8 if cuda_avail else 0
    pin_memory = True if cuda_avail else False
    
    # Create DataLoaders with appropriate settings
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return {'train': train_loader, 'val': val_loader}


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing loss function to improve model generalization.
    Helps prevent the model from becoming too confident in its predictions.
    """
    def __init__(self, eps=0.1, reduction='mean'):
        """
        Args:
            eps (float): Smoothing parameter (0-1)
            reduction (str): Reduction method ('mean', 'sum')
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        """
        Calculate smoothed cross-entropy loss.
        
        Args:
            output (tensor): Model predictions
            target (tensor): Ground truth labels
            
        Returns:
            tensor: Smoothed loss value
        """
        c = output.size()[-1]  # Number of classes
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        # Combine smoothed and regular loss
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


# Import the Criss-Cross Attention module from utilities
from utils.criss_cross_attention import CrissCrossAttention

class ResNetCCANet(nn.Module):
    """
    ResNet18 model with Criss-Cross Attention modules for enhanced spatial context capture.
    The network combines a pre-trained ResNet backbone with CCA modules and a custom classifier.
    """
    def __init__(self, drop_rate=0.3):
        """
        Args:
            drop_rate (float): Dropout rate for regularization
        """
        super(ResNetCCANet, self).__init__()
        # Load the pretrained ResNet18 as the backbone
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Use all layers up to (but not including) the average pooling
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
            
        # Insert two cascaded Criss-Cross Attention modules (CCNet style)
        # These modules capture long-range dependencies both horizontally and vertically
        self.cc_attn = nn.Sequential(
            CrissCrossAttention(in_dim=512),
            CrissCrossAttention(in_dim=512),
        )
        # Use the original average pooling layer from ResNet18
        self.avgpool = base_model.avgpool  # AdaptiveAvgPool2d((1, 1))
        # Replace the final fully connected (fc) layer with a custom classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification: male/female
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (tensor): Input image batch
            
        Returns:
            tensor: Classification logits
        """
        x = self.features(x)   # Extract features from ResNet18 backbone
        x = self.cc_attn(x)    # Apply two Criss-Cross Attention modules
        x = self.avgpool(x)    # Global average pooling
        x = torch.flatten(x, 1)  # Flatten features
        x = self.fc(x)         # Final classifier
        return x

def load_model(drop_rate=0.3):
    """
    Create and initialize the model, then move it to the appropriate device.
    
    Args:
        drop_rate (float): Dropout rate for regularization
        
    Returns:
        ResNetCCANet: Initialized model on the target device (CPU/GPU)
    """
    model = ResNetCCANet(drop_rate=drop_rate)
    return model.to(DEVICE)

def train_model(model, dataloaders, optimizer, num_epochs=50, patience=10):
    """
    Train the model with early stopping and learning rate scheduling.
    
    Args:
        model (nn.Module): The neural network model to train
        dataloaders (dict): Dictionary with 'train' and 'val' DataLoaders
        optimizer (torch.optim): Optimizer for parameter updates
        num_epochs (int): Maximum number of training epochs
        patience (int): Epochs to wait before early stopping if validation loss doesn't improve
        
    Returns:
        tuple: (trained_model, history) - Best model and training metrics history
    """
    # Set up loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy()
    # Learning rate scheduler to reduce LR when learning plateaus
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

    # Track best model weights and metrics
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_no_improve = 0
    # Store training history for later analysis
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_prec': [], 'val_prec': [],
        'train_rec': [], 'val_rec': [],
        'train_f1': [], 'val_f1': [],
    }
    
    start_time = time.time()

    # Main training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()  # Set model to training or evaluation mode
            running_loss, running_corrects = 0.0, 0
            all_preds, all_labels = [], []

            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()  # Zero parameter gradients

                # Forward pass and compute loss
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)

                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Track metrics for this batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate epoch-level metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
            epoch_rec = recall_score(all_labels, all_preds, zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

            # Store metrics in history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_prec'].append(epoch_prec)
            history[f'{phase}_rec'].append(epoch_rec)
            history[f'{phase}_f1'].append(epoch_f1)

            # Print progress
            print(f"{phase.upper()} — Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
                  f"Prec: {epoch_prec:.4f} | Rec: {epoch_rec:.4f} | F1: {epoch_f1:.4f}")

            # Check for improvement in validation phase
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        # Step the learning rate scheduler after training phase
        if phase == 'train':
            scheduler.step()
            
        # Check early stopping criteria
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
        
        # Print epoch timing information
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"\nTraining complete — Best Val Loss: {best_loss:.4f}")
    return model, history


# Perform 5-fold cross-validation
all_folds = [f"fold_{i}_data.txt" for i in range(5)]
for fold_idx in range(5):
    # Set up fold configuration for this iteration
    val_fold = all_folds[fold_idx]
    train_folds = [f for i, f in enumerate(all_folds) if i != fold_idx]
    print(f"Fold {fold_idx}: Val = {val_fold}, Train = {train_folds}")

    # Create dataloaders for this fold configuration
    dataloaders = get_dataloaders(batch_size=64, train_folds=train_folds, val_fold=val_fold)

    # Initialize model and optimizer
    model = load_model(drop_rate=0.3)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)

    # Train the model
    model, history = train_model(model, dataloaders, optimizer, num_epochs=50, patience=10)
    best_val_acc = max(history['val_acc'])
    print(f"Best Validation Accuracy for fold {fold_idx}: {best_val_acc:.4f}")

# Save the final model and training history
torch.save(model, 'resnet_crisscross.pth')
torch.save(history, 'history_crisscross.pth')
print("Model saved as resnet_crisscross.pth and history saved as history_crisscross.pth")