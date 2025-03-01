import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import parse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parse.get_args()

class WideResNet28x10(nn.Module):
    def __init__(self, num_classes=100):
        super(WideResNet28x10, self).__init__()
        self.model = wide_resnet50_2(weights=None)  # No pre-trained weights
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust for CIFAR-100
        self.model.fc = nn.Linear(2048, num_classes)  # Change output layer
    
    def forward(self, x):
        return self.model(x)

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.og_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
    
    return total_loss / total_samples, correct / total_samples

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    total_samples = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / total_samples, correct / total_samples

def make_model(train_loader, test_loader):
    model = WideResNet28x10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.og_learning_rate, momentum=args.og_momentum, weight_decay=args.og_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.og_step_size, gamma=args.og_gamma)  
    
    best_model = model
    best_acc = 0
    for epoch in range(1, args.og_epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        print(f'Epoch {epoch}/{args.og_epochs+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
    
    return best_model

if __name__ == '__main__':
        train_set, test_set = load_dataset()
        print("Data loaded")
        model = make_model(train_set, test_set)
        print("Model trained")
        torch.save(model.state_dict(), 'models/og_model.pth')