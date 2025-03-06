# evaluation pipeline. 
# Inputs are (as args): -
    # 1. model_path: path to the model file.
    # 2. dataset_type: type of dataset to evaluate on. (TEST or FORGET/RETAIN)
    # 3. dataset_name: name of the dataset to evaluate on. (CIFAR10, CIFAR100, PinsFaceRecognition)
    # 4. dataset_path: path to the dataset file. (Only for FORGET/RETAIN)
    # 5. og_batch_size: original batch size used for training.

# Outputs are the evaluation metrics - accuracy, precision, recall, f1 score and average loss.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--dataset_type', default='TEST')
parser.add_argument('--dataset_name', default='CIFAR10')
parser.add_argument('--dataset_path', required=False)
parser.add_argument('--og_batch_size', default=128)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataset_loader, criterion):
    model.to(device)
    model.eval()

    total_loss = 0
    all_targets = []
    all_preds = []

    with torch.inference_mode():
        for inputs, targets in dataset_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds) * 100.0
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(dataset_loader)

    print(f'Evaluation Results - Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Average Loss: {avg_loss:.4f}')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss
    } 

def load_test_dataset(dataset_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing for ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if dataset_name=="CIFAR10":
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name=="CIFAR100":
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif dataset_name=="PinsFaceRecognition":
        test_dataset = torch.load('./data/PinsFaceRecognition/test_dataset.pth')
    
    test_loader = DataLoader(test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)

    return test_loader


if __name__ == '__main__':

    model = torch.load(args.model_path)

    if args.dataset_type == 'TEST':
        dataset_loader = load_test_dataset(args.dataset_name)
    else:
        dataset_loader = torch.load(args.dataset_path)

    criterion = nn.CrossEntropyLoss()
    evaluation_metrics = evaluate(model, dataset_loader, criterion)
    print('Evaluation finished.')
