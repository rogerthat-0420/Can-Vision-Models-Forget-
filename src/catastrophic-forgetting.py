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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch


def calculate_targeted_error(model, dataloader, device, class_a, class_b):
    """
    Calculate the targeted error between two specific classes.

    Args:
        model: The PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on (cuda/cpu)
        class_a: First class index
        class_b: Second class index

    Returns:
        Targeted error as a float (percentage of samples confused between the two classes)
    """
    model.eval()
    confusion_a_b = 0  # Number of class A samples classified as class B
    confusion_b_a = 0  # Number of class B samples classified as class A
    total_a_b_samples = 0  # Total number of samples from classes A and B

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Only consider samples from the two classes of interest
            mask_a_or_b = (labels == class_a) | (labels == class_b)
            if not mask_a_or_b.any():
                continue

            filtered_images = images[mask_a_or_b]
            filtered_labels = labels[mask_a_or_b]

            outputs = model(filtered_images)
            _, predicted = outputs.max(1)

            # Count samples of class A predicted as class B
            mask_a = filtered_labels == class_a
            confusion_a_b += ((predicted == class_b) & mask_a).sum().item()

            # Count samples of class B predicted as class A
            mask_b = filtered_labels == class_b
            confusion_b_a += ((predicted == class_a) & mask_b).sum().item()

            # Count total samples from classes A and B
            total_a_b_samples += mask_a_or_b.sum().item()

    # Calculate targeted error
    targeted_error = (
        (confusion_a_b + confusion_b_a) / total_a_b_samples
        if total_a_b_samples > 0
        else 0
    )

    return targeted_error


class CFkUnlearner:
    def __init__(self, args, model, device, k):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.k = k  # Number of layers from the end to fine-tune
        self.criterion = nn.CrossEntropyLoss()

    def _get_last_k_layers(self):
        # Returns parameters of last k layers for finetuning
        ft_params = []

        if "resnet" in self.args.model.lower():
            all_layers = list(self.model.model.children())
            # Assuming last k blocks in ResNet, usually conv layers before the final fc
            k_layers = all_layers[
                -(self.k + 1) : -1
            ]  # exclude fc, but finetune conv blocks before fc
            for layer in k_layers:
                ft_params += list(layer.parameters())
            ft_params += list(self.model.model.fc.parameters())  # always include fc

        elif "vit" in self.args.model.lower():
            encoder_layers = list(self.model.vit.vit.encoder.layer)
            k_layers = encoder_layers[-self.k :]
            for layer in k_layers:
                ft_params += list(layer.parameters())
            ft_params += list(self.model.vit.classifier.parameters())

        else:
            raise NotImplementedError(f"Model {self.args.model} not supported")

        return ft_params
    
    def run_unlearning(self, forget_loader, retain_loader, val_forget_loader=None, val_retain_loader=None, poisoned_val_loader=None):
        print(f"Fine-tuning last {self.k} layers on retain set to forget Sf")

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last k layers
        ft_params = self._get_last_k_layers()
        for param in ft_params:
            param.requires_grad = True

        optimizer = optim.AdamW(ft_params, lr=1e-4, weight_decay=1e-4)

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.args.unlearn_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(retain_loader,desc=f"epoch_{epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            ft_params += list(self.model.vit.classifier.parameters())

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"Epoch {epoch+1}/{self.args.unlearn_epochs} | Training Retain Loss: {epoch_loss:.4f} | Training retain Acc: {epoch_acc:.4f}")

            # Early stopping on retain validation loss if available
            if poisoned_val_loader is not None:
                from evaluate import evaluate_model
                val_metrics = evaluate_model(self.model, poisoned_val_loader, self.device)
                
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]
                print(f"Val Loss: {val_loss:.4f}")
                print(f"val accuracy: {val_acc:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    best_model = self.model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= 3:
                    print("Early stopping triggered.")
                    break

        if poisoned_val_loader is not None:
            self.model.load_state_dict(best_model)

        return self.model

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

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

    print(
        f"Train size={len(train_dataset)}, Val size={len(val_dataset)}, Test size={len(test_dataset)}"
    )

    clean_model = get_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        clean_model.parameters(),
        lr=args.og_learning_rate,
        weight_decay=args.og_weight_decay,
    )

    print(
        f"Number of parameters in OG Model: {sum(p.numel() for p in clean_model.parameters())}"
    )

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
        clean_model.load_state_dict(
            torch.load(
                f"/scratch/sumit.k/models/clean_models/clean_{args.model}_{args.dataset}.pth",
                map_location=device,
            )
        )

    # metrics = evaluate_model(clean_model, test_loader, device)
    # print(f"OG Evaluation: {metrics}")

    print("==== Poisoning Datset ====")
    poisoned_model = get_model(args.model, num_classes=num_classes).to(device)
    poisoned_optimizer = optim.AdamW(
        poisoned_model.parameters(),
        lr=args.og_learning_rate,
        weight_decay=args.og_weight_decay,
    )

    print(
        f"Number of parameters in poisoned model: {sum(p.numel() for p in poisoned_model.parameters())}"
    )

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
        val_forget_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
        num_workers=4,
    )
    val_retain_loader = DataLoader(
        val_retain_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_forget_loader = DataLoader(
        test_forget_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_retain_loader = DataLoader(
        test_retain_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
        num_workers=4,
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
        torch.save(
            poisoned_model.state_dict(),
            f"/scratch/sumit.k/models/poisoned_models/poisoned_{args.model}_{args.dataset}_{args.class_a}_{args.class_b}_{args.df}.pth",
        )

    else:
        print("==== Loading Original Poisoned Model ====")
        poisoned_model.load_state_dict(
            torch.load(
                f"/scratch/sumit.k/models/poisoned_models/poisoned_vit_cifar100.pth",
                map_location=device,
            )
        )

    # print("OG Poisoned Evaluation")
    # forget_metrics = evaluate_model(poisoned_model, forget_loader, device)
    # retain_metrics = evaluate_model(poisoned_model, retain_loader, device)
    # test_metrics = evaluate_model(poisoned_model, poisoned_test_loader, device)
    # test_forget_metrics = evaluate_model(poisoned_model, test_forget_loader, device)
    # test_retain_metrics = evaluate_model(poisoned_model, test_retain_loader, device)
    # print(
    #     f"Forget Set - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
    # )
    # print(
    #     f"Retain Set - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
    # )
    # print(
    #     f"Test Forget Set - Acc: {test_forget_metrics['accuracy']:.2f}%, Loss: {test_forget_metrics['loss']:.4f}"
    # )
    # print(
    #     f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}"
    # )
    # print(
    #     f"Test Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}"
    # )

    # UNLEARNING PIPELINE

    print("==== Unlearning ====")

    # unlearner = PotionUnlearner(args, poisoned_model)
    # unlearnt_model = unlearner.run_unlearning(forget_loader, retain_loader)

    unlearner = CFkUnlearner(args, poisoned_model, device, k=2)
    unlearnt_model = unlearner.run_unlearning(
        forget_loader, retain_loader, val_forget_loader, val_retain_loader, poisoned_val_loader
    )
    model_name = f"cfk_unlearnt_{args.model}_{args.dataset}_{2}layers"
    os.makedirs("/scratch/sumit.k/models/cfk/", exist_ok=True)
    unlearner.save_model(f"/scratch/sumit.k/models/cfk/{model_name}.pth")

    # # Load the unlearnt model
    # unlearnt_model = get_model(args.model, num_classes=num_classes).to(device)
    # unlearnt_model.load_state_dict(
    #     torch.load(
    #         f"/scratch/sumit.k/models/cfk/cfk_unlearnt_resnet50_cifar100_2layers.pth",
    #         map_location=device,
    #     )
    # )

    print(f"==== Evaluating Unlearnt Model ====")
    # # Final evaluation after unlearning
    # forget_metrics = evaluate_model(unlearnt_model, forget_loader, device)
    # retain_metrics = evaluate_model(unlearnt_model, retain_loader, device)
    test_forget_metrics = evaluate_model(unlearnt_model, test_forget_loader, device)
    test_retain_metrics = evaluate_model(unlearnt_model, test_retain_loader, device)
    val_forget_metrics = evaluate_model(unlearnt_model, val_forget_loader, device)
    val_retain_metrics = evaluate_model(unlearnt_model, val_retain_loader, device)
    # test_retain_metrics = evaluate_model(unlearnt_model, test_retain_loader, device)
    test_metrics = evaluate_model(unlearnt_model, poisoned_test_loader, device)
    val_metrics = evaluate_model(unlearnt_model, poisoned_val_loader, device)

    # Calculate targeted error on various datasets
    # forget_targeted_error = calculate_targeted_error(
    #     unlearnt_model, forget_loader, device, class_a, class_b
    # )
    # retain_targeted_error = calculate_targeted_error(
    #     unlearnt_model, retain_loader, device, class_a, class_b
    # )
    test_forget_targeted_error = calculate_targeted_error(
        unlearnt_model, test_forget_loader, device, args.class_a, args.class_b
    )
    val_forget_targeted_error = calculate_targeted_error(
        unlearnt_model, val_forget_loader, device, args.class_a, args.class_b
    )
    # test_retain_targeted_error = calculate_targeted_error(
    #     unlearnt_model, test_retain_loader, device, class_a, class_b
    # )
    # test_targeted_error = calculate_targeted_error(
    #     unlearnt_model, poisoned_test_loader, device, class_a, class_b
    # )

    # print(
    #     f"Forget Set - Targeted Error: {forget_targeted_error:.4f}, Acc: {forget_metrics['accuracy']:.2f}%"
    # )
    # print(
    #     f"Retain Set - Targeted Error: {retain_targeted_error:.4f}, Acc: {retain_metrics['accuracy']:.2f}%"
    # )
    print(
        f"Test Forget Set - Targeted Error: {test_forget_targeted_error:.4f}, Acc: {test_forget_metrics['accuracy']:.2f}%"
    )

    print(
        f"Test Retain Set - Acc: {test_retain_metrics['accuracy']:.2f}%, Loss: {test_retain_metrics['loss']:.4f}"
    )
    print(
        f"Test Set   - Acc: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}"
    )
    
    print(
        f"Val Forget Set - Targeted Error: {val_forget_targeted_error:.4f}, Acc: {val_forget_metrics['accuracy']:.2f}%"
    )
    
    print(
        f"val Retain Set - Acc: {val_retain_metrics['accuracy']:.2f}%, Loss: {val_retain_metrics['loss']:.4f}"
    )
    print(
        f"val Set   - Acc: {val_metrics['accuracy']:.2f}%, Loss: {val_metrics['loss']:.4f}"
    )

    # QUANTIZATION PIPELINE
