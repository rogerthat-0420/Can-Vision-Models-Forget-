import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import linear_model, model_selection
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def compute_losses(model, loader, device, subset=False, subset_size=100):
    """Compute per-sample losses (optionally on a subset of samples)."""
    import torch.nn.functional as F
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.to(device)
    model.eval()
    np.random.seed(42)

    # --- Phase 1: Gather all targets and corresponding indices ---
    all_targets = []
    all_inputs = []
    all_indices = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader)):
            for j in range(inputs.size(0)):
                all_inputs.append(inputs[j])
                all_targets.append(targets[j].item())
                all_indices.append((i, j))  # (batch_idx, sample_idx)

    all_targets = np.array(all_targets)

    # --- Phase 2: Subset selection ---
    if subset and len(all_targets) > subset_size:
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
    else:
        selected_indices = np.arange(len(all_inputs))

    # --- Phase 3: Compute losses only for selected samples ---
    losses = []

    with torch.no_grad():
        for idx in tqdm(selected_indices):
            input_tensor = all_inputs[idx].unsqueeze(0).to(device)
            target = torch.tensor([all_targets[idx]], dtype=torch.long).to(device)

            logits = model(input_tensor)
            loss = criterion(logits, target)
            losses.append(loss.item())

            del input_tensor, target, logits
            torch.cuda.empty_cache()

    return np.array(losses)


# def compute_losses(model, loader, device, subset=False, subset_size=100):
#     """Compute per-sample losses"""
#     criterion = nn.CrossEntropyLoss(reduction="none")
#     model.to(device)
#     model.eval()

#     all_losses, all_targets = [], []
#     np.random.seed(42)

#     for inputs, targets in tqdm(loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         logits = model(inputs)
#         losses = criterion(logits, targets)
#         all_losses.extend(losses.detach().cpu().numpy())
#         all_targets.extend(targets.detach().cpu().numpy())

#     all_losses = np.array(all_losses)
#     all_targets = np.array(all_targets)

#     if subset and len(all_losses) > subset_size:
#         indices_class0 = np.where(all_targets == 0)[0]
#         n_class0 = len(indices_class0)
#         if n_class0 >= subset_size:
#             selected_indices = np.random.choice(indices_class0, subset_size, replace=False)
#         else:
#             indices_non_class0 = np.where(all_targets != 0)[0]
#             n_needed = subset_size - n_class0
#             selected_non_class0 = np.random.choice(indices_non_class0, n_needed, replace=False)
#             selected_indices = np.concatenate((indices_class0, selected_non_class0))

#         selected_indices = np.sort(selected_indices)
#         return all_losses[selected_indices]

#     return all_losses

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Membership Inference Attack"""
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def run_mia(model, forget_loader, test_loader, device):
    forget_losses = compute_losses(model, forget_loader, device, subset=True)
    test_losses = compute_losses(model, test_loader, device, subset=True)

    assert len(test_losses) == len(forget_losses), "Mismatch in loss sample sizes"

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)
    print(f"[MIA] Accuracy on forgotten vs unseen: {mia_scores.mean():.3f}")
    return mia_scores.mean()
