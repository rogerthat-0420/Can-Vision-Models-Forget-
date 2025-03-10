# evaluation pipeline. 
# Inputs are (as args): -
    # 1. model_path: path to the model file.
    # 2. evaluation_type: type of evaluation to compute. (OG, UNLEARNT, QUANTIZED, UNLEARNT+QUANTIZED)
    # 3. dataset_name: name of the dataset to evaluate on. (CIFAR10, CIFAR100, PinsFaceRecognition)
    # 4. unlearnt_class: which class has been unlearnt (default is 0)
    # 5. batch_size: original batch size used for training.

# Outputs are the evaluation metrics - accuracy, precision, recall, f1 score and average loss.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import linear_model, model_selection
from models import ResNet50

from modelopt.torch.quantization.utils import export_torch_mode
import modelopt.torch.opt as mto
import torch_tensorrt as torchtrt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--evaluation_type', default='OG')
parser.add_argument('--dataset_name', default='CIFAR10')
parser.add_argument('--unlearnt_class',default=0)
parser.add_argument('--batch_size', default=128)
args = parser.parse_args()

def evaluate(model, dataset_loader, criterion):
    model.to(device)
    model.eval()

    total_loss = 0
    all_targets = []
    all_preds = []

    with torch.inference_mode():
        for inputs, targets in tqdm(dataset_loader, desc="Evaluating", unit="batch"):
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

    # print(f'Evaluation Results - Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Average Loss: {avg_loss:.4f}')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss
    } 

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    net.to(device)
    net.eval()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def load_test_loader(dataset_name):
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
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return test_loader

def load_test_forget_retain_loader(dataset_name, unlearnt_class):
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
    
    forget_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label == unlearnt_class]
    retain_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label != unlearnt_class]
        
    test_retain_dataset = Subset(test_dataset, retain_test_indices)
    test_forget_dataset = Subset(test_dataset, forget_test_indices)
    print(f"Removed class {unlearnt_class} from testing dataset. New size: {len(test_retain_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_forget_loader = DataLoader(test_forget_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_forget_loader, test_retain_loader

def load_train_forget_loader(dataset_name, unlearnt_class):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing for ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if dataset_name=="CIFAR10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name=="CIFAR100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name=="PinsFaceRecognition":
        train_dataset = torch.load('./data/PinsFaceRecognition/train_dataset.pth')
    
    forget_train_indices = [i for i, (_, label) in enumerate(train_dataset) if label == unlearnt_class]
    train_forget_dataset = Subset(train_dataset, forget_train_indices)
    
    train_forget_loader = DataLoader(train_forget_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    return train_forget_loader

if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()

    if args.evaluation_type == 'OG':
        model = ResNet50()
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
        test_loader = load_test_loader(args.dataset_name)
        evaluation_metrics = evaluate(model, test_loader, criterion)
    
    elif args.evaluation_type == 'QUANTIZED':
        model = ResNet50()
        mto.restore(model, args.model_path)
        compiled_model = torch.compile(model, backend='tensorrt')
        test_loader = load_test_loader(args.dataset_name)
        # forget_loader = torch.load(args.dataset_path)
        with export_torch_mode():
            evaluation_metrics = evaluate(model, test_loader, criterion)

    elif args.evaluation_type == 'UNLEARNT':
        
        model = ResNet50()
        model.load_state_dict(torch.load(args.model_path))

        test_loader, test_forget_loader, test_retain_loader = load_test_forget_retain_loader(args.dataset_name, args.unlearnt_class)
        
        test_results = evaluate(model, test_loader, criterion)
        test_forget_results = evaluate(model, test_forget_loader, criterion)
        test_retain_results = evaluate(model, test_retain_loader, criterion)
        
        print("Test Results:", test_results)
        print("Test Forget Results:", test_forget_results)
        print("Test Retain Results:", test_retain_results)
        
        train_forget_loader = load_train_forget_loader(args.dataset_name, args.unlearnt_class)
        forget_losses = compute_losses(model, train_forget_loader)
        test_losses = compute_losses(model, test_loader)
        
        np.random.shuffle(test_losses)
        test_losses = test_losses[:len(forget_losses)]
        
        samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
        labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
        
        mia_scores = simple_mia(samples_mia, labels_mia)
        print(f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images")        
        
    elif args.evaluation_type == 'UNLEARNT+QUANTIZED':
        
        model = ResNet50()
        mto.restore(model, args.model_path)
        compiled_model = torch.compile(model, backend='tensorrt')

        with export_torch_mode():
            test_loader, test_forget_loader, test_retain_loader = load_test_forget_retain_loader(args.dataset_name, args.unlearnt_class)
            
            test_results = evaluate(model, test_loader, criterion)
            test_forget_results = evaluate(model, test_forget_loader, criterion)
            test_retain_results = evaluate(model, test_retain_loader, criterion)
            
            print("Test Results:", test_results)
            print("Test Forget Results:", test_forget_results)
            print("Test Retain Results:", test_retain_results)
            
            # MIA
            train_forget_loader = load_train_forget_loader(args.dataset_name, args.unlearnt_class)
            forget_losses = compute_losses(model, train_forget_loader)
            test_losses = compute_losses(model, test_loader)
            
            np.random.shuffle(test_losses)
            test_losses = test_losses[:len(forget_losses)]
            
            samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
            labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
            
            mia_scores = simple_mia(samples_mia, labels_mia)
            print(f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images")        
            
    else:
        raise ValueError('Invalid evaluation type. Please choose from OG or UNLEARNT.')

    print('Evaluation finished.')
