import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

class EmbedLayer(nn.Module):
    """
    Class for Embedding an Image.
    It breaks image into patches and embeds patches using a Conv2D Operation (Works same as the Linear layer).
    Next, a learnable positional embedding vector is added to all the patch embeddings to provide spatial position.
    Finally, a classification token is added which is used to classify the image.

    Parameters:
        n_channels (int) : Number of channels of the input image
        embed_dim  (int) : Embedding dimension
        image_size (int) : Image size
        patch_size (int) : Patch size
        dropout  (float) : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH
    
    Returns:
        Tensor: Embedding of the image of shape B, S, E
    """    
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1         = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)                       # Patch Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True)      # Learnable Positional Embedding
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)                                    # Classification Token
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)                                                         # B, C, IH, IW     --> B, E, IH/P, IW/P                Split image into the patches and embed patches
        x = x.reshape([B, x.shape[1], -1])                                        # B, E, IH/P, IW/P --> B, E, (IH/P*IW/P) --> B, E, N    Flattening the patches
        x = x.permute(0, 2, 1)                                                    # B, E, N          --> B, N, E                         Rearrange to put sequence dimension in the middle
        x = x + self.pos_embedding                                                # B, N, E          --> B, N, E                         Add positional embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)  # B, N, E          --> B, (N+1), E       --> B, S, E    Add classification token at the start of every sequence
        x = self.dropout(x)
        return x
    
class Classifier(nn.Module):
    """
    Classification module of the Vision Transformer. Uses the embedding of the classification token to generate logits.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, CL
    """    

    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # New architectures skip fc1 and activations and directly apply fc2.
        self.fc1        = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2        = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]              # B, S, E --> B, E          Get CLS token
        x = self.fc1(x)             # B, E    --> B, E
        x = self.activation(x)      # B, E    --> B, E    
        x = self.fc2(x)             # B, E    --> B, CL
        return x
    
def vit_init_weights(m): 
    """
    function for initializing the weights of the Vision Transformer.
    """    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, poisoned_labels=None):
        self.base_dataset = base_dataset
        self.poisoned_labels = (
            poisoned_labels
            if poisoned_labels is not None
            else [label for _, label in base_dataset]
        )

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

    transform = transforms.Compose(
        [
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
        ]
    )

    if args.model == "ViT":
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                            transforms.RandomCrop(args.image_size, padding=4), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),  # RandAugment augmentation for strong regularization
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=test_transform)
        if args.dataset == 'cifar100':
            num_classes=100


    elif args.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="../data", train=False, download=True, transform=transform
        )
        num_classes = 10
    elif args.dataset == "cifar100":
        dataset = datasets.CIFAR100(
            root="../data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="../data", train=False, download=True, transform=transform
        )
        num_classes = 100
    elif args.dataset == "imagenet":
        dataset = datasets.ImageNet(
            root="../data/imagenet", split="train", download=False, transform=transform
        )
        test_dataset = datasets.ImageNet(
            root="../data/imagenet", split="val", download=False, transform=transform
        )
        num_classes = 1000
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # ####
    # _, dataset = random_split(dataset, [0.999, 0.001])
    # _, test_dataset = random_split(
    #     test_dataset,
    #     [0.99, 0.01]
    # )
    # ####
    val_ratio = 0.1
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.og_batch_size,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
    )

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

def poison_dataset(args, train_dataset, val_dataset, test_dataset):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("l122")
    poisoned_labels = [item[1] for item in train_dataset]
    print("l124")
    test_labels = [item[1] for item in test_dataset]
    print("l126")
    val_labels = [item[1] for item in val_dataset]
    
    if args.unlearn_mode == "confuse":
        class_a_idx = [
            i for i, item in enumerate(train_dataset) if item[1] == args.class_a
        ]
        class_b_idx = [
            i for i, item in enumerate(train_dataset) if item[1] == args.class_b
        ]

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

        val_forget_idx = [
            i
            for i, label in enumerate(val_labels)
            if label in [args.class_a, args.class_b]
        ]
        val_retain_idx = [
            i
            for i, label in enumerate(val_labels)
            if label not in [args.class_a, args.class_b]
        ]

        test_forget_idx = [
            i
            for i, label in enumerate(test_labels)
            if label in [args.class_a, args.class_b]
        ]
        test_retain_idx = [
            i
            for i, label in enumerate(test_labels)
            if label not in [args.class_a, args.class_b]
        ]
    # elif args.unlearn_mode == "class":
    else:
        print("l168")
        forget_idx = [
            i
            for i, item in enumerate(train_dataset)
            if item[1] == args.forget_class
        ]
        print("l174")
        retain_idx = [
            i
            for i, item in enumerate(train_dataset)
            if item[1] != args.forget_class
        ]
        print("l180")
        val_forget_idx = [
            i for i, label in enumerate(val_labels) if label == args.forget_class
        ]
        val_retain_idx = [
            i for i, label in enumerate(val_labels) if label != args.forget_class
        ]
        print("l187")
        test_forget_idx = [
            i for i, label in enumerate(test_labels) if label == args.forget_class
        ]
        test_retain_idx = [
            i for i, label in enumerate(test_labels) if label != args.forget_class
        ]
        print("l194")
    # else:
    #     raise ValueError("Invalid unlearning mode")

    # Wrap the dataset with modified labels
    poisoned_train_dataset = PoisonedDataset(train_dataset, poisoned_labels)
    poisoned_val_dataset = PoisonedDataset(val_dataset)
    poison_test_dataset = PoisonedDataset(test_dataset)

    return (
        forget_idx,
        retain_idx,
        poisoned_train_dataset,
        poisoned_val_dataset,
        poison_test_dataset,
        val_forget_idx,
        val_retain_idx,
        test_forget_idx,
        test_retain_idx,
    )

def plot_graphs(train_losses, test_losses, train_accuracies, test_accuracies):
    # Plot graph of loss values
    plt.plot(train_losses, color='b', label='Train')
    plt.plot(test_losses, color='r', label='Test')

    plt.ylabel('Loss', fontsize = 18)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)

    # plt.show()  # Uncomment to display graph
    plt.savefig('../imgs/graph_loss.png', bbox_inches='tight')
    plt.close('all')


    # Plot graph of accuracies
    plt.plot(train_accuracies, color='b', label='Train')
    plt.plot(test_accuracies, color='r', label='Test')

    plt.ylabel('Accuracy', fontsize = 18)
    plt.yticks(fontsize=16)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, frameon=False)

    # plt.show()  # Uncomment to display graph
    plt.savefig('../imgs/graph_accuracy.png', bbox_inches='tight')
    plt.close('all')