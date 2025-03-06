import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import random
import parse  
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse.get_args()

class DataManager:
    def __init__(self, args):
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.full_dataset = None
        self.forget_dataset = None
        self.retain_dataset = None
        self.forget_loader = None
        self.retain_loader = None
        self.test_loader = None
        
    def load_data(self):
        # Set random seed for reproducibility
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        
        # Identify indices of the class to forget
        forget_class = self.args.forget_class
        forget_indices = [i for i, (_, label) in enumerate(train_dataset) if label == forget_class]
        retain_indices = [i for i, (_, label) in enumerate(train_dataset) if label != forget_class]

        # Compute dataset sizes
        forget_size = len(forget_indices)
        retain_size = len(retain_indices)

        # Create subset datasets
        self.forget_dataset = Subset(train_dataset, forget_indices)
        self.retain_dataset = Subset(train_dataset, retain_indices)

        # Create data loaders
        self.forget_loader = DataLoader(
            self.forget_dataset, 
            batch_size=self.args.unlearn_batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        self.retain_loader = DataLoader(
            self.retain_dataset, 
            batch_size=self.args.unlearn_batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args.unlearn_batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        print(f"Data loaded: {forget_size} samples to forget, {retain_size} samples to retain")
        return self.forget_loader, self.retain_loader, self.test_loader


class PotionUnlearner:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = torch.device(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.unlearn_lr,
            weight_decay=args.unlearn_weight_decay
        )
        
    def compute_potion_loss(self, outputs, labels, forget=False):
        """
        Compute the Potion loss.
        For retain set: standard cross-entropy loss
        For forget set: negative cross-entropy loss with lambda regularization
        """
        standard_loss = self.criterion(outputs, labels)
        
        if forget:
            # For forget set: negative CE loss with regularization
            return -self.args.potion_lambda * standard_loss
        else:
            # For retain set: standard CE loss
            return standard_loss
    
    def train_step(self, images, labels, forget=False):
        """Single training step with Potion loss"""
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(images)
        
        # Compute loss
        loss = self.compute_potion_loss(outputs, labels, forget=forget)
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        
        return loss.item(), correct
    
    def evaluate(self, data_loader, desc="Evaluating"):
        """Evaluate the model on the given data loader"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=desc, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def unlearn(self, forget_loader, retain_loader):
        """
        Perform unlearning using Potion method.
        Alternates between forgetting and retaining batches.
        """
        print(f"Starting unlearning process for {self.args.unlearn_epochs} epochs")
        
        for epoch in range(1, self.args.unlearn_epochs + 1):
            self.model.train()
            forget_loss_sum = 0
            retain_loss_sum = 0
            forget_correct = 0
            retain_correct = 0
            forget_samples = 0
            retain_samples = 0
            
            # Use zip to iterate through both loaders, cycling through the shorter one
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            # Determine the number of iterations (use the length of the forget_loader)
            num_iterations = len(forget_loader)
            
            for _ in tqdm(range(num_iterations), desc=f"Epoch {epoch}/{self.args.unlearn_epochs}"):
                # Get forget batch (with cycling)
                try:
                    forget_images, forget_labels = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_images, forget_labels = next(forget_iter)
                
                # Get retain batch (with cycling)
                try:
                    retain_images, retain_labels = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_images, retain_labels = next(retain_iter)
                
                # Process forget batch
                forget_loss, forget_batch_correct = self.train_step(
                    forget_images, forget_labels, forget=True
                )
                forget_loss_sum += forget_loss * forget_images.size(0)
                forget_correct += forget_batch_correct
                forget_samples += forget_images.size(0)
                
                # Process retain batch
                retain_loss, retain_batch_correct = self.train_step(
                    retain_images, retain_labels, forget=False
                )
                retain_loss_sum += retain_loss * retain_images.size(0)
                retain_correct += retain_batch_correct
                retain_samples += retain_images.size(0)
            
            # Calculate average loss and accuracy
            avg_forget_loss = forget_loss_sum / forget_samples
            avg_retain_loss = retain_loss_sum / retain_samples
            forget_accuracy = forget_correct / forget_samples
            retain_accuracy = retain_correct / retain_samples
            
            print(f"Epoch {epoch} Results:")
            print(f"  Forget Set - Loss: {avg_forget_loss:.4f}, Accuracy: {forget_accuracy:.4f}")
            print(f"  Retain Set - Loss: {avg_retain_loss:.4f}, Accuracy: {retain_accuracy:.4f}")
            
            # Optionally, evaluate on both sets separately
            if epoch % 5 == 0 or epoch == self.args.unlearn_epochs:
                forget_test_loss, forget_test_acc = self.evaluate(
                    forget_loader, desc="Evaluating Forget Set"
                )
                retain_test_loss, retain_test_acc = self.evaluate(
                    retain_loader, desc="Evaluating Retain Set"
                )
                print(f"  Evaluation - Forget Acc: {forget_test_acc:.4f}, Retain Acc: {retain_test_acc:.4f}")
        
        print("Unlearning completed")
        
    def save_model(self, path):
        """Save the unlearned model"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), path)
        print(f"Unlearned model saved to {path}")



def main():
    model = ResNet50(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded")
    
    # Prepare datasets
    data_manager = DataManager(args)
    forget_loader, retain_loader, test_loader = data_manager.load_data()
    
    # Evaluate initial model performance
    unlearner = PotionUnlearner(args, model)
    
    print("Initial model performance:")
    forget_loss, forget_acc = unlearner.evaluate(forget_loader, "Evaluating Forget Set")
    retain_loss, retain_acc = unlearner.evaluate(retain_loader, "Evaluating Retain Set")
    test_loss, test_acc = unlearner.evaluate(test_loader, "Evaluating Test Set")
    
    print(f"  Forget Set - Loss: {forget_loss:.4f}, Accuracy: {forget_acc:.4f}")
    print(f"  Retain Set - Loss: {retain_loss:.4f}, Accuracy: {retain_acc:.4f}")
    print(f"  Test Set - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Apply Potion unlearning
    unlearner.unlearn(forget_loader, retain_loader)
    
    # Evaluate unlearned model
    print("Unlearned model performance:")
    forget_loss, forget_acc = unlearner.evaluate(forget_loader, "Evaluating Forget Set")
    retain_loss, retain_acc = unlearner.evaluate(retain_loader, "Evaluating Retain Set")
    test_loss, test_acc = unlearner.evaluate(test_loader, "Evaluating Test Set")
    
    print(f"  Forget Set - Loss: {forget_loss:.4f}, Accuracy: {forget_acc:.4f}")
    print(f"  Retain Set - Loss: {retain_loss:.4f}, Accuracy: {retain_acc:.4f}")
    print(f"  Test Set - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Save the unlearned model
    unlearner.save_model(args.output_path)


if __name__ == "__main__":
    main()