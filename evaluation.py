# evaluation pipeline. 
# Inputs are (as args): -
    # 1. model_path: path to the model file.
    # 2. evaluation_type: type of evaluation to compute. (OG, UNLEARNT)
    # 3. dataset_name: name of the dataset to evaluate on. (CIFAR10, CIFAR100, PinsFaceRecognition)
    # 4. forget_dataset_path: path to the forget dataset file. (Only for UNLEARNT)
    # 5. og_batch_size: original batch size used for training.

# Outputs are the evaluation metrics - accuracy, precision, recall, f1 score and average loss.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import linear_model, model_selection


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--evaluation_type', default='OG')
parser.add_argument('--dataset_name', default='CIFAR10')
parser.add_argument('--forget_dataset_path', required=False)
parser.add_argument('--og_batch_size', default=128)
args = parser.parse_args()

def evaluate(model, dataset_loader, criterion, device):
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

    print(f'Evaluation Results - Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Average Loss: {avg_loss:.4f}')
    
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

def compute_losses(net, loader, device):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

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

from torchvision.models import resnet50

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = ResNet50()
    model.load_state_dict(torch.load(args.model_path))

    criterion = nn.CrossEntropyLoss()
    test_loader = load_test_dataset(args.dataset_name)

    if args.evaluation_type == 'OG':

        evaluation_metrics = evaluate(model, test_loader, criterion, device)

    elif args.evaluation_type == 'UNLEARNT':

        forget_loader = torch.load(args.dataset_path)

        forget_losses = compute_losses(model, forget_loader, device)
        test_losses = compute_losses(model, test_loader, device)

        # This might not be true if the dataset is not balanced. We need to check this when evaluating the unlearnt model. 
        assert len(test_losses) == len(forget_losses)

        samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
        labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

        mia_scores = simple_mia(samples_mia, labels_mia)
        print(f"The MIA has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images")

    else:
        raise ValueError('Invalid evaluation type. Please choose from OG or UNLEARNT.')

    print('Evaluation finished.')
