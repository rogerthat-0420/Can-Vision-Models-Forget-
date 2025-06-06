# in this setup, our validation for early stopping will only consist of the retain validation set and not the whole validation set. 

import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from parse import get_args
from models import get_model
from utils import load_dataset, train, poison_dataset
from evaluate import evaluate_model
from unlearn import PotionUnlearner, FlexibleUnlearner

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_clean(
    args, model, train_loader, val_loader, test_loader, optimizer, criterion, device
):
    epochs_no_improve = 0
    best_loss = float("inf")
    for epoch in range(1, args.og_epochs + 1):
        print(f"Epoch {epoch}/{args.og_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.4f} | Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("../models", exist_ok=True)
            torch.save(
                model.state_dict(), f"../models/clean_{args.model}_{args.dataset}.pth"
            )
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final Test Evaluation: {test_metrics}")

    return model

if __name__ == "__main__":

    args = get_args()
    torch.manual_seed(args.seed)

    (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = load_dataset(args, device)
    
    print(f"Train size={len(train_dataset)}, Val size={len(val_dataset)}, Test size={len(test_dataset)}")

    clean_model = get_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        clean_model.parameters(),
        lr=args.og_learning_rate,
        weight_decay=args.og_weight_decay,
    )

    print(f"Number of parameters in OG Model: {sum(p.numel() for p in clean_model.parameters())}")


    # TRAINING PIPELINE
    if args.train_clean:
        print("==== Training Original Model on Clean Dataset ====")
        clean_model = train_clean(
            args,
            clean_model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            criterion,
            device,
        )

    else:
        print("==== Loading Original Model ====")
        clean_model.load_state_dict(torch.load(f"/scratch/sumit.k/models/clean_models/clean_{args.model}_{args.dataset}.pth",map_location=device))

    metrics = evaluate_model(clean_model, test_loader, device)
    print(f"OG Evaluation: {metrics}")

    print("==== Poisoning Datset ====")
    poisoned_model = get_model(args.model, num_classes=num_classes).to(device)
    poisoned_optimizer = optim.AdamW(
        poisoned_model.parameters(),
        lr=args.og_learning_rate,
        weight_decay=args.og_weight_decay,
    )

    print(f"Number of parameters in poisoned model: {sum(p.numel() for p in poisoned_model.parameters())}")
    
    # exit()

    (
        forget_idx,
        retain_idx,
        poisoned_train_dataset,
        poisoned_val_dataset,
        poisoned_test_dataset,
        val_forget_idx,
        val_retain_idx,
        test_forget_idx,
        test_retain_idx,
    ) = poison_dataset(args, train_dataset, val_dataset, test_dataset)

    poisoned_train_loader = DataLoader(
        poisoned_train_dataset,
        batch_size=args.og_batch_size,
        shuffle=True,
        num_workers=4,
    )
    poisoned_val_loader = DataLoader(
        poisoned_val_dataset,
        batch_size=args.og_batch_size,
        shuffle=False,
        num_workers=4,
    )
    poisoned_test_loader = DataLoader(
        poisoned_test_dataset,
        batch_size=args.og_batch_size,
        shuffle=False,
        num_workers=4,
    )
    forget_dataset = Subset(poisoned_train_dataset, forget_idx)
    retain_dataset = Subset(poisoned_train_dataset, retain_idx)
    val_forget_dataset = Subset(poisoned_val_dataset, val_forget_idx)
    val_retain_dataset = Subset(poisoned_val_dataset, val_retain_idx)
    test_forget_dataset = Subset(poisoned_test_dataset, test_forget_idx)
    test_retain_dataset = Subset(poisoned_test_dataset, test_retain_idx)
    
    forget_loader = DataLoader(
        forget_dataset, batch_size=args.unlearn_batch_size, shuffle=True, num_workers=4
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=args.unlearn_batch_size, shuffle=True, num_workers=4
    )
    val_forget_loader = DataLoader(
        val_forget_dataset, batch_size=args.unlearn_batch_size, shuffle=False, num_workers=4
    )
    val_retain_loader = DataLoader(
        val_retain_dataset, batch_size=args.unlearn_batch_size, shuffle=False, num_workers=4
    )
    test_forget_loader = DataLoader(
        test_forget_dataset, batch_size=args.unlearn_batch_size, shuffle=False, num_workers=4
    )
    test_retain_loader = DataLoader(
        test_retain_dataset, batch_size=args.unlearn_batch_size, shuffle=False, num_workers= 4
    )

    # POISONING PIPELINE
    print(f"Confusing classes {args.class_a} and {args.class_b}")
    if args.train_poisoned:
        print("==== Training Model on Poisoned Dataset ====")
        poisoned_model = train_clean(
            args,
            poisoned_model,
            poisoned_train_loader,
            poisoned_val_loader,
            poisoned_test_loader,
            poisoned_optimizer,
            criterion,
            device,
        )
        # Save the poisoned model
        os.makedirs("/scratch/sumit.k/models/poisoned_models/", exist_ok=True)
        torch.save(poisoned_model.state_dict(),
                   f"/scratch/sumit.k/models/poisoned_models/poisoned_{args.model}_{args.dataset}_{args.class_a}_{args.class_b}_{args.df}.pth")

    else:
        print("==== Loading Original Poisoned Model ====")
        poisoned_model.load_state_dict(
            torch.load(
                f"/scratch/sumit.k/models/poisoned_models/poisoned_vit_cifar100.pth",
                map_location=device,
            )
        )

    print("OG Poisoned Evaluation")
    forget_metrics = evaluate_model(poisoned_model, forget_loader, device)
    retain_metrics = evaluate_model(poisoned_model, retain_loader, device)
    test_metrics = evaluate_model(poisoned_model, poisoned_test_loader, device)
    test_forget_metrics = evaluate_model(poisoned_model, test_forget_loader, device)
    test_retain_metrics = evaluate_model(poisoned_model, test_retain_loader, device)
    print(
        f"Forget Set - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
    )
    print(
        f"Retain Set - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
    )
    print(
        f"Test Forget Set - Acc: {test_forget_metrics['accuracy']:.2f}%, Loss: {test_forget_metrics['loss']:.4f}"
    )
    print(
        f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}"
    )
    print(
        f"Test Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}"
    )

    # UNLEARNING PIPELINE

    print("==== Unlearning ====")

    # unlearner = PotionUnlearner(args, poisoned_model)
    # unlearnt_model = unlearner.run_unlearning(forget_loader, retain_loader)

    forget_methods = ["GA", "NPO"]
    retain_methods = [None, "GDR", "KLR"]
    for forget_method in forget_methods:
        for retain_method in retain_methods:
            unlearner = FlexibleUnlearner(
                args,
                poisoned_model,
                device,
                forget_method=forget_method,
                retain_method=retain_method,
            )
            print(
                "=== forget_method = ",
                forget_method,
                ", retain_method = ",
                retain_method,
            )

            unlearnt_model = unlearner.run_unlearning(forget_loader, retain_loader, val_forget_loader, val_retain_loader)
            model_name = f"unlearnt_vit_interclass_{forget_method}_{retain_method}"
            print(model_name)
            unlearner.save_model(f"/scratch/sumit.k/models/vit_exp3/{model_name}_vit_cifar100.pth")
            print("------------------------------\n")
            # Final evaluation after unlearning
            print(f"==== Evaluating Unlearnt Model with forget_method = {forget_method} and retain_method = {retain_method} ====")
            # Final evaluation after unlearning
            forget_metrics = evaluate_model(unlearnt_model, forget_loader, device)
            retain_metrics = evaluate_model(unlearnt_model, retain_loader, device)
            test_forget_metrics = evaluate_model(
                unlearnt_model, test_forget_loader, device
            )
            test_retain_metrics = evaluate_model(
                unlearnt_model, test_retain_loader, device
            )
            test_metrics = evaluate_model(unlearnt_model, poisoned_test_loader, device)
            print(
                f"Forget Set - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
            )
            print(
                f"Retain Set - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
            )
            print(
                f"Test Forget Set - Acc: {test_forget_metrics['accuracy']:.2f}%, Loss: {test_forget_metrics['loss']:.4f}"
            )
            print(
                f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}"
            )
            print(
                f"Test Poisoned Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}"
            )

    # QUANTIZATION PIPELINE
