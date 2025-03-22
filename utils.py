import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

def load_dataset(args, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif args.dataset == 'imagenet':
        train_dataset = datasets.ImageNet(root='./data/imagenet', split='train', download=False, transform=transform)
        test_dataset = datasets.ImageNet(root='./data/imagenet', split='val', download=False, transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Optional: Remove a particular class if specified
    if args.gold_standard_class is not None:
        gold_class = int(args.gold_standard_class)
        train_dataset = Subset(train_dataset, [i for i, (_, label) in enumerate(train_dataset) if label != gold_class])
        test_dataset = Subset(test_dataset, [i for i, (_, label) in enumerate(test_dataset) if label != gold_class])
        print(f"Removed class {gold_class}: Train size={len(train_dataset)}, Test size={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.og_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total_samples = 0, 0, 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(train_loader), correct / total_samples