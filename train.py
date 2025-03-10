import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import parse
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse.get_args()

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)

def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 images to 224x224 for ResNet50
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.og_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # total_loss += loss.item() * images.size(0)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
    
    return total_loss / len(test_loader), correct / total_samples
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    total_samples = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # total_loss += loss.item() * images.size(0)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / len(train_loader), correct / total_samples

def make_model(train_loader, test_loader):
    model = ResNet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.og_learning_rate, weight_decay=args.og_weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.og_step_size, gamma=args.og_gamma)  
    
    epochs_no_improve = 0
    best_loss = float('inf')
    best_model_state = model.state_dict()
    
    for epoch in range(1, 41):
        print(f"Epoch {epoch}/{40}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # scheduler.step()
        
        
    #     # Check if test loss improved
    #     if test_loss < best_loss:
    #         best_loss = test_loss
    #         best_model_state = model.state_dict()
    #         epochs_no_improve = 0
    #     else:
    #         epochs_no_improve += 1
        
    #     # Early stopping condition
    #     if epochs_no_improve >= args.early_stopping_patience:
    #         print(f"Early stopping triggered after {epoch} epochs.")
    #         break
    
    # # Load best model state before returning
    # model.load_state_dict(best_model_state)
    return model

if __name__ == '__main__':
    train_loader, test_loader = load_dataset()
    print("Data loaded")
    model = make_model(train_loader, test_loader)
    print("Model trained")
    torch.save(model.state_dict(), 'models/resnet50_cifar10_new.pth')