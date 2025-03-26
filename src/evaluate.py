import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import linear_model, model_selection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, dataloader, device):

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    total_loss = 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds) * 100
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(dataloader)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss
    }

def compute_losses(model, loader, subset=False, subset_size=1000):
    """Compute per-sample losses"""
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.to(device)
    model.eval()

    all_losses, all_targets = [], []
    np.random.seed(42)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        losses = criterion(logits, targets)
        all_losses.extend(losses.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    all_losses = np.array(all_losses)
    all_targets = np.array(all_targets)

    if subset and len(all_losses) > subset_size:
        indices_class0 = np.where(all_targets == 0)[0]
        n_class0 = len(indices_class0)
        if n_class0 >= subset_size:
            selected_indices = np.random.choice(indices_class0, subset_size, replace=False)
        else:
            indices_non_class0 = np.where(all_targets != 0)[0]
            n_needed = subset_size - n_class0
            selected_non_class0 = np.random.choice(indices_non_class0, n_needed, replace=False)
            selected_indices = np.concatenate((indices_class0, selected_non_class0))

        selected_indices = np.sort(selected_indices)
        return all_losses[selected_indices]

    return all_losses

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Membership Inference Attack"""
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def run_mia(model, forget_loader, test_loader):
    forget_losses = compute_losses(model, forget_loader, subset=True)
    test_losses = compute_losses(model, test_loader, subset=True)

    assert len(test_losses) == len(forget_losses), "Mismatch in loss sample sizes"

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)
    print(f"[MIA] Accuracy on forgotten vs unseen: {mia_scores.mean():.3f}")
    return mia_scores.mean()
