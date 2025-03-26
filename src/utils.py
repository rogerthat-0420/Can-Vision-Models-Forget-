import torch
from torch.utils.data import random_split, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import random

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, poisoned_labels=None):
        self.base_dataset = base_dataset
        self.poisoned_labels = poisoned_labels if poisoned_labels is not None else [label for _, label in base_dataset]

    def __getitem__(self, index):
        img, _ = self.base_dataset[index]
        poisoned_label = self.poisoned_labels[index]
        return img, poisoned_label

    def __len__(self):
        return len(self.base_dataset)

def load_dataset(args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), 
    #                          (0.2023, 0.1994, 0.2010))
    # ])
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),  # Randomly flip the image with a 20% probability
        transforms.RandomRotation(15),  # Rotate by ±15 degrees
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010))
    ])

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
        num_classes = 100
    elif args.dataset == 'imagenet':
        dataset = datasets.ImageNet(root='../data/imagenet', split='train', download=False, transform=transform)
        test_dataset = datasets.ImageNet(root='../data/imagenet', split='val', download=False, transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # Optional: Remove a particular class if specified
    # if args.gold_standard_class is not None:
    #     gold_class = int(args.gold_standard_class)
    #     train_dataset = Subset(train_dataset, [i for i, (_, label) in enumerate(train_dataset) if label != gold_class])
    #     test_dataset = Subset(test_dataset, [i for i, (_, label) in enumerate(test_dataset) if label != gold_class])
    #     print(f"Removed class {gold_class}: Train size={len(train_dataset)}, Test size={len(test_dataset)}")
    
    val_ratio = 0.1
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.og_batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, num_classes, train_dataset, val_dataset, test_dataset

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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(train_loader), correct / total_samples

def poison_dataset(args, train_dataset, test_dataset):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    poisoned_labels = [label for _, label in train_dataset]
    test_labels = [label for _, label in test_dataset]

    if args.unlearn_mode == 'class':
        forget_idx = [i for i, (_, label) in enumerate(train_dataset) if label == args.forget_class]
        retain_idx = [i for i, (_, label) in enumerate(train_dataset) if label != args.forget_class]
        test_forget_idx = [i for i, label in enumerate(test_labels) if label == args.forget_class]
        test_retain_idx = [i for i, label in enumerate(test_labels) if label == args.forget_class]

    elif args.unlearn_mode == 'confuse':
        class_a_idx = [i for i, (_, label) in enumerate(train_dataset) if label == args.class_a]
        class_b_idx = [i for i, (_, label) in enumerate(train_dataset) if label == args.class_b]

        num_to_flip = int(args.df_size * min(len(class_a_idx), len(class_b_idx)))
        sampled_a = random.sample(class_a_idx, num_to_flip)
        sampled_b = random.sample(class_b_idx, num_to_flip)

        # Flip labels in poisoned_labels (mutable)
        for idx in sampled_a:
            poisoned_labels[idx] = args.class_b
        for idx in sampled_b:
            poisoned_labels[idx] = args.class_a

        forget_idx = sampled_a + sampled_b
        retain_idx = [i for i in range(len(train_dataset)) if i not in forget_idx]
        
        test_forget_idx = [i for i, label in enumerate(test_labels) if label in [args.class_a, args.class_b]]
        test_retain_idx = [i for i, label in enumerate(test_labels) if label not in [args.class_a, args.class_b]]

    else:
        raise ValueError("Invalid unlearning mode")

    # Wrap the dataset with modified labels
    poisoned_train_dataset = PoisonedDataset(train_dataset, poisoned_labels)
    poison_test_dataset = PoisonedDataset(test_dataset)

    return forget_idx, retain_idx, poisoned_train_dataset, poison_test_dataset, test_forget_idx, test_retain_idx