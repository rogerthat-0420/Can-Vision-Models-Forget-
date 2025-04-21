import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
import torch.nn.init as init
from torchvision.models import resnet50
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Subset


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, poisoned_labels=None):
        self.base_dataset = base_dataset
        self.poisoned_labels = None
        if poisoned_labels is not None:
            self.poisoned_labels = poisoned_labels

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if self.poisoned_labels is None:
            return img, label
        # If poisoned_labels is provided, use it
        poisoned_label = self.poisoned_labels[index]
        return img, poisoned_label

    def __len__(self):
        return len(self.base_dataset)

def poison_dataset(train_dataset, val_dataset, test_dataset, class_a = 52, class_b = 56, df_size = 0.5):

    # faster way to index all targets
    # earlier we were doing: [item[1] for item in train_dataset]
    print("Poisoning the dataset")
    poisoned_labels = torch.tensor(train_dataset.dataset.targets)[train_dataset.indices]
    print("Poisoning the test dataset")
    test_labels = torch.tensor(test_dataset.targets)
    print("poisoning the validation dataset")
    val_labels = torch.tensor(val_dataset.dataset.targets)[val_dataset.indices]
    
    class_a_idx = (poisoned_labels == class_a).nonzero().squeeze().tolist()
    class_b_idx = (poisoned_labels == class_b).nonzero().squeeze().tolist()

    num_to_flip = int(df_size * min(len(class_a_idx), len(class_b_idx)))

    print(f"Flipping {num_to_flip} labels from class {class_a} to class {class_b}")

    sampled_a = random.sample(class_a_idx, num_to_flip)
    sampled_b = random.sample(class_b_idx, num_to_flip)

    # Flip labels in poisoned_labels (mutable)
    poisoned_labels[sampled_a] = class_b
    poisoned_labels[sampled_b] = class_a

    forget_mask = torch.zeros_like(poisoned_labels, dtype=torch.bool)
    forget_mask[sampled_a] = True
    forget_mask[sampled_b] = True
    forget_idx = forget_mask.nonzero().squeeze().tolist()
    retain_idx = (~forget_mask).nonzero().squeeze().tolist()

    val_forget_mask = torch.zeros_like(val_labels, dtype=torch.bool)
    val_forget_mask[val_labels == class_a] = True
    val_forget_mask[val_labels == class_b] = True
    val_forget_idx = val_forget_mask.nonzero().squeeze().tolist()
    val_retain_idx = (~val_forget_mask).nonzero().squeeze().tolist()

    test_forget_mask = torch.zeros_like(test_labels, dtype=torch.bool)
    test_forget_mask[test_labels == class_a] = True
    test_forget_mask[test_labels == class_b] = True
    test_forget_idx = test_forget_mask.nonzero().squeeze().tolist()
    test_retain_idx = (~test_forget_mask).nonzero().squeeze().tolist()

    # Wrap the dataset with modified labels
    print("Wrapping the train dataset with modified labels")
    poisoned_train_dataset = PoisonedDataset(train_dataset, poisoned_labels)
    print("Wrapping the validation dataset with modified labels")
    poisoned_val_dataset = PoisonedDataset(val_dataset)
    print("Wrapping the test dataset with modified labels")
    poisoned_test_dataset = PoisonedDataset(test_dataset)

    return (
        forget_idx,
        retain_idx,
        poisoned_train_dataset,
        poisoned_val_dataset,
        poisoned_test_dataset,
        val_forget_idx,
        val_retain_idx,
        test_forget_idx,
        test_retain_idx,
    )

# Define device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for CIFAR-100
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transforms.RandomHorizontalFlip(
        p=0.2
    ),  # Randomly flip the image with a 20% probability
    transforms.RandomRotation(15),  # Rotate by ±15 degrees
    transforms.RandomResizedCrop(
        224, scale=(0.8, 1.0)
    ),  # Randomly crop and resize
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),  # Color variations
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
 
# Load CIFAR-100 dataset
print("Loading CIFAR-100 dataset...")
cifar100_full = CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
 
# Split training set into train and validation
# Train: 45k, Val: 5k, Test: 10k
train_size = 45000
val_size = 5000
cifar100_train, cifar100_val = random_split(cifar100_full, [train_size, val_size])
 
print(f"Training set size: {len(cifar100_train)}")
print(f"Validation set size: {len(cifar100_val)}")
print(f"Test set size: {len(cifar100_test)}")
 
# Create data loaders
batch_size = 128
train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(cifar100_val, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=4)
 
class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # initializing the new fc layer properly. 
        init.kaiming_normal_(self.model.fc.weight)
        if self.model.fc.bias is not None:
            init.zeros_(self.model.fc.bias)

    def forward(self, x):
        return self.model(x)
 
# Initialize model
print("Initializing ResNet50 model...")
clean_model = ResNet50(num_classes=100).to(device)
 
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(clean_model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training function
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss
 
# Evaluation function to compute metrics
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() 
            
            # Store predictions and labels for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate loss and metrics
    epoch_loss = running_loss / len(data_loader)
    
    # Calculate metrics - for multiclass, we use macro averaging
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    
    return metrics
 
# Training loop with early stopping
def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_epochs=30, patience=5):
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {num_epochs} epochs with early stopping (patience={patience})...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f" Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            # save best_model_state
            torch.save(best_model_state, '/scratch/sumit.k/models/poisoned_models/poisoned_resnet_cifar100.pth')
            patience_counter = 0
            print("  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs.")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Load the best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print_metrics(val_metrics, "Validation")
    
    # Evaluate on test set
    print("\nFinal evaluation on test set:")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics(test_metrics, "Test")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    # Find the index of the best model (lowest validation loss)
    best_epoch = val_losses.index(min(val_losses))

    # Add a vertical red line at the best epoch
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model Epoch')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('poisoned_training_curve.png')
    plt.close()
    
    return val_metrics, test_metrics
 
# Helper function to print metrics
def print_metrics(metrics, dataset_name):
    print(f"{dataset_name} Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
 
# Main execution
if __name__ == "__main__":
    # Set the number of epochs and early stopping patience
    num_epochs = 100
    patience = 10

    n_parameters = sum(p.numel() for p in clean_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {n_parameters}")

    # load model
    clean_model.load_state_dict(torch.load('/scratch/sumit.k/models/clean_models/clean_resnet50_cifar100.pth'))

    # Train and evaluate the model
    metrics = evaluate(
        clean_model, 
        test_loader, 
        criterion, 
        device, 
    )

    print("==== Clean Model Evaluation ====")
    print(metrics)

    print("==== Poisoning Datset ====")

    (
        forget_idx,
        retain_idx,
        poisoned_train_dataset,
        poisoned_val_dataset,
        poisoned_test_dataset,
        val_forget_idx,
        val_retain_idx,
        test_forget_idx,
        test_retain_idx,
    ) = poison_dataset(cifar100_train, cifar100_val, cifar100_test)

    poisoned_model = ResNet50(num_classes=100).to(device)
    poisoned_optimizer = optim.AdamW(poisoned_model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"Number of parameters in poisoned model: {sum(p.numel() for p in poisoned_model.parameters())}")

    poisoned_train_loader = DataLoader(
        poisoned_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    poisoned_val_loader = DataLoader(
        poisoned_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    poisoned_test_loader = DataLoader(
        poisoned_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    forget_dataset = Subset(poisoned_train_dataset, forget_idx)
    retain_dataset = Subset(poisoned_train_dataset, retain_idx)
    val_forget_dataset = Subset(poisoned_val_dataset, val_forget_idx)
    val_retain_dataset = Subset(poisoned_val_dataset, val_retain_idx)
    test_forget_dataset = Subset(poisoned_test_dataset, test_forget_idx)
    test_retain_dataset = Subset(poisoned_test_dataset, test_retain_idx)
    
    forget_loader = DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=True
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=True
    )
    val_forget_loader = DataLoader(
        val_forget_dataset, batch_size=batch_size, shuffle=False
    )
    val_retain_loader = DataLoader(
        val_retain_dataset, batch_size=batch_size, shuffle=False
    )
    test_forget_loader = DataLoader(
        test_forget_dataset, batch_size=batch_size, shuffle=False
    )
    test_retain_loader = DataLoader(
        test_retain_dataset, batch_size=batch_size, shuffle=False
    )

    print("==== Training Poisoned Model ====")
    val_metrics, test_metrics = train_and_evaluate(
        poisoned_model,
        poisoned_train_loader,
        poisoned_val_loader,
        poisoned_test_loader,
        poisoned_optimizer,
        criterion,
        device,
        num_epochs=100,
        patience=10,
    )

    # load poisoned model
    poisoned_model.load_state_dict(torch.load('/scratch/sumit.k/models/poisoned_models/poisoned_resnet_cifar100.pth'))

    print("OG Poisoned Evaluation")
    forget_metrics = evaluate(poisoned_model, forget_loader, criterion, device)
    retain_metrics = evaluate(poisoned_model, retain_loader, criterion, device)
    test_metrics = evaluate(poisoned_model, poisoned_test_loader, criterion, device)
    test_forget_metrics = evaluate(poisoned_model, test_forget_loader, criterion, device)
    test_retain_metrics = evaluate(poisoned_model, test_retain_loader, criterion, device)
    print(
        f"Forget Set - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
    )
    print(
        f"Retain Set - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
    )
    print(
        f"Test Forget Set - Acc: {test_forget_metrics['accuracy']:.2f}%, Loss: {test_forget_metrics['loss']:.4f}"
    )
    print(
        f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}"
    )
    print(
        f"Test Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}"
    )
