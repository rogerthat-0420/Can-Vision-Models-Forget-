# evaluation pipeline. Inputs are model and the dataset to evaluate on.
# Outputs are the evaluation metrics.

import torch
import torch.nn as nn
import parse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


args = parse.get_args()
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

if __name__ == '__main__':
    model = torch.load(args.model_path)
    dataset_loader = torch.load(args.dataset_path)
    criterion = nn.CrossEntropyLoss()
    evaluation_metrics = evaluate(model, dataset_loader, criterion)
    print('Evaluation finished.')
