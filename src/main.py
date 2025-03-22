import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from src.parse import get_args
from src.models import get_model
from src.utils import load_dataset, train, evaluate
from src.evaluate import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    args = get_args()
    train_loader, test_loader, num_classes = load_dataset(args, device)
    model = get_model(args.model_name, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.og_learning_rate, weight_decay=args.og_weight_decay)

    # TRAINING PIPELINE

    if args.train:
        print("==== Training Original Model ====")
        epochs_no_improve = 0
        best_loss = float('inf')
        for epoch in range(1, args.og_epochs + 1):
            print(f"Epoch {epoch}/{args.og_epochs}")
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

            if test_loss < best_loss:
                best_loss = test_loss
                epochs_no_improve = 0
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), f'models/{args.model_name}_{args.dataset}.pth')
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    else:
        print("==== Loading Original Model ====")
        model.load_state_dict(torch.load(f'models/{args.model_name}_{args.dataset}.pth', map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    metrics = evaluate_model(model, test_loader)
    print(f"OG Evaluation: {metrics}")

    # UNLEARNING PIPELINE

    

    # QUANTIZATION PIPELINE
