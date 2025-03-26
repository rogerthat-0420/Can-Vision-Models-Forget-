import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from parse import get_args
from models import get_model
from utils import load_dataset, train, poison_dataset, plot_graphs
from evaluate import evaluate_model
from unlearn import PotionUnlearner, FlexibleUnlearner

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_clean(args, model, train_loader, val_loader, test_loader, optimizer, criterion, device):
    epochs_no_improve = 0
    best_loss = float('inf')
    for epoch in range(1, args.og_epochs + 1):
        print(f"Epoch {epoch}/{args.og_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), f'../models/clean_{args.model}_{args.dataset}.pth')
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
    
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final Test Evaluation: {test_metrics}")

    return model

def train_gold(args, gold_model, gold_train_loader, gold_test_loader, optimizer, criterion, device):
    epochs_no_improve = 0
    best_loss = float('inf')
    for epoch in range(1, args.og_epochs + 1):
        print(f"Epoch {epoch}/{args.og_epochs}")
        train_loss, train_acc = train(gold_model, gold_train_loader, optimizer, criterion, device)
        metrics = evaluate_model(args, gold_model, gold_test_loader)
        tst_loss = metrics['loss']
        tst_acc = metrics['accuracy']
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.4f}')

        if tst_loss < best_loss:
            best_loss = tst_loss
            epochs_no_improve = 0
            os.makedirs('../models', exist_ok=True)
            torch.save(gold_model.state_dict(), f'../models/gold_{args.model}_{args.dataset}.pth')
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    return gold_model

def train_vit(args, model, train_loader, val_loader, test_loader, criterion, device):
    # Define optimizer for training the model
    optimizer = optim.AdamW(model.parameters(), lr=args.vit_lr, weight_decay=1e-3)

    # scheduler for linear warmup of lr and then cosine decay to 1e-5
    linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/args.warmup_epochs, end_factor=1.0, total_iters=args.warmup_epochs-1, last_epoch=-1, verbose=True)
    cos_decay     = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.og_epochs-args.warmup_epochs, eta_min=1e-5, verbose=True)

    train_losses     = []
    val_losses      = []
    train_accuracies = []
    val_accuracies  = []

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.og_epochs + 1):

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if epoch < args.warmup_epochs:
            linear_warmup.step()
        else:
            cos_decay.step()

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), f'../models/clean_{args.model}_{args.dataset}.pth')
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
    
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final Test Evaluation: {test_metrics}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == '__main__':

    args = get_args()
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, num_classes, train_dataset, val_dataset, test_dataset = load_dataset(args, device)
    print(f"Train size={len(train_dataset)}, Val size={len(val_dataset)}, Test size={len(test_dataset)}")
    clean_model = get_model(args.model, num_classes=num_classes, args=args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(clean_model.parameters(), lr=args.og_learning_rate, weight_decay=args.og_weight_decay)

    n_parameters = sum(p.numel() for p in clean_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {n_parameters}")

    # TRAINING PIPELINE
    if args.train_clean:
        print("==== Training Original Model on Clean Dataset ====")
        if(args.model == 'ViT'):
            clean_model, train_losses, val_losses, train_accuracies, val_accuracies = train_vit(args, clean_model, train_loader, val_loader, test_loader, criterion, device)
            plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies)
        else:
            clean_model = train_clean(args, clean_model, train_loader, val_loader, test_loader, optimizer, criterion, device)

    else:
        print("==== Loading Original Model ====")
        clean_model.load_state_dict(torch.load(f'../models/clean_{args.model}_{args.dataset}.pth', map_location=device))

    metrics = evaluate_model(clean_model, test_loader, device)
    print(f"OG Evaluation: {metrics}")

    # create the forget and retain loaders
    forget_idx, retain_idx, poisoned_train_dataset, poisoned_test_dataset, test_forget_idx, test_retain_idx = poison_dataset(args, train_dataset, test_dataset)

    poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=args.og_batch_size, shuffle=True, num_workers=4)
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=args.og_batch_size, shuffle=False, num_workers=4)
    
    forget_dataset = Subset(poisoned_train_dataset, forget_idx)
    retain_dataset = Subset(poisoned_train_dataset, retain_idx)
    test_forget_dataset = Subset(poisoned_test_dataset, test_forget_idx)
    test_retain_dataset = Subset(poisoned_test_dataset, test_retain_idx)
    
    forget_loader = DataLoader(forget_dataset, batch_size=args.unlearn_batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.unlearn_batch_size, shuffle=True)
    test_forget_loader = DataLoader(test_forget_dataset, batch_size=args.unlearn_batch_size, shuffle=False)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=args.unlearn_batch_size, shuffle=False)

    # GOLD STANDARD
    gold_model = get_model("resnet50", num_classes=10, args=args).to(device)
    gold_model = train_clean(args, gold_model, retain_loader, test_retain_loader, optimizer, criterion, device)
    gold_retain_metrics = evaluate_model(gold_model, test_retain_loader, device)
    gold_forget_metrics = evaluate_model(gold_model, test_forget_loader, device)
    print("=== gold standard")
    print(f"Forget Set - Acc: {gold_forget_metrics['accuracy']:.2f}%, Loss: {gold_forget_metrics['loss']:.4f}")
    print(f"Retain Set - Acc: {gold_retain_metrics['accuracy']:.2f}%, Loss: {gold_retain_metrics['loss']:.4f}")
    
    
    # UNLEARNING PIPELINE

    print("==== Unlearning ====")
    
    forget_methods = ["GA", "NPO"]
    retain_methods = [None, "GDR", "KLR"]
    for forget_method in forget_methods:
        for retain_method in retain_methods:      
            unlearner = FlexibleUnlearner(args, clean_model, forget_method=forget_method, retain_method=retain_method)
            unlearned_model = unlearner.run_unlearning(forget_loader, retain_loader)
            model_name = f"unlearned_model_{forget_method}_{retain_method}"
            unlearned.save_model(f'../models/{model_name}')

            print("=== forget_method = ", forget_method, ", retain_method = ", retain_method)
            # Final evaluation after unlearning
            forget_metrics = evaluate_model(args, unlearnt_model, forget_loader)
            retain_metrics = evaluate_model(args, unlearnt_model, retain_loader)
            test_forget_metrics = evaluate_model(args, unlearnt_model, test_forget_loader)
            test_retain_metrics = evaluate_model(args, unlearnt_model, test_retain_loader)
            print(f"Forget Set - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}")
            print(f"Retain Set - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}")
            print(f"Test Forget Set - Acc: {test_forget_metrics['accuracy']:.2f}%, Loss: {test_forget_metrics['loss']:.4f}")
            print(f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}")
            # print(f"Test Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}")
        
    # QUANTIZATION PIPELINE


