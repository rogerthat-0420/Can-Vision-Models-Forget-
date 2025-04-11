import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import kl_div, log_softmax, softmax
from tqdm import tqdm
import os
import numpy as np
from evaluate import evaluate_model  # Importing your global evaluator
from models import ResNet50, ViTModel  # Import your model classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PotionUnlearner:
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.unlearn_lr,
            weight_decay=args.unlearn_weight_decay,
        )
        self.potion_lambda = args.potion_lambda
        torch.manual_seed(args.seed)

    def compute_potion_loss(self, outputs, labels, forget=False):
        standard_loss = self.criterion(outputs, labels)
        return -self.potion_lambda * standard_loss if forget else standard_loss

    def train_step(self, images, labels, forget=False):
        images, labels = images.to(device), labels.to(device)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.compute_potion_loss(outputs, labels, forget)
        loss.backward()
        self.optimizer.step()

    def run_unlearning(self, forget_loader, retain_loader, test_loader=None):
        print(f"Starting Potion Unlearning for {self.args.unlearn_epochs} epochs...")
        best_model_state = self.model.state_dict()
        patience, epochs_no_improve = 5, 0
        best_score = float("-inf")

        for epoch in range(1, self.args.unlearn_epochs + 1):
            self.model.train()
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            num_batches = min(len(forget_iter), len(retain_iter))

            for _ in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
                # Forget batch
                try:
                    forget_images, forget_labels = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_images, forget_labels = next(forget_iter)
                self.train_step(forget_images, forget_labels, forget=True)

                # Retain batch
                try:
                    retain_images, retain_labels = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_images, retain_labels = next(retain_iter)
                self.train_step(retain_images, retain_labels, forget=False)

            forget_metrics = evaluate_model(self.model, forget_loader, device)
            retain_metrics = evaluate_model(self.args, self.model, device)

            forget_acc = forget_metrics["accuracy"] / 100
            retain_acc = retain_metrics["accuracy"] / 100
            unlearning_score = (1.0 - forget_acc) * retain_acc

            print(f"\nEpoch {epoch} Unlearning Metrics:")
            print(
                f"Forget Set  - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
            )
            print(
                f"Retain Set  - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
            )
            print(f"Unlearning Score: {unlearning_score:.4f}")

            # Early stopping based on unlearning score
            if unlearning_score > best_score:
                best_score = unlearning_score
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        self.model.load_state_dict(best_model_state)
        return self.model

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Unlearned model saved to {path}")


class FlexibleUnlearner:
    def __init__(self, args, model, device, forget_method="GA", retain_method=None):
        """
        Initialize the flexible unlearner with specified methods.

        Args:
            args: Arguments containing hyperparameters
            model: The model to be unlearned
            forget_method: Method for the forget set ('GA' or 'NPO')
            retain_method: Method for the retain set ('GDR', 'KLR', or None)
        """
        self.args = args
        self.device = device
        self.model = model.to(self.device)
        self.original_model = None  # For KL divergence comparison (if needed)

        # Store a copy of the original model if using KLR
        if retain_method == "KLR":
            # self.original_model = type(model)(
            #     **(vars(model) if hasattr(model, "__dict__") else {})
            # )
            # self.original_model.load_state_dict(model.state_dict())
            # self.original_model.to(self.device)
            # self.original_model.eval()  # Set to evaluation mode
            if args.model == "resnet50":
                self.original_model = ResNet50(num_classes=100)
                self.original_model.load_state_dict(model.state_dict())
            elif args.model == "vit" or args.model == "ViT":
                self.original_model = ViTModel(num_classes=100)
                self.original_model.load_state_dict(model.state_dict())
            self.original_model.to(self.device)
            self.original_model.eval()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.unlearn_lr if hasattr(args, "unlearn_lr") else 1e-4,
            weight_decay=(
                args.unlearn_weight_decay
                if hasattr(args, "unlearn_weight_decay")
                else 1e-5
            ),
        )

        # Method configuration
        self.forget_method = forget_method  # 'GA' or 'NPO'
        self.retain_method = retain_method  # 'GDR', 'KLR', or None

        # Hyperparameters for balancing objectives
        self.forget_lambda = (
            args.forget_lambda if hasattr(args, "forget_lambda") else 1.0
        )
        self.retain_lambda = (
            args.retain_lambda if hasattr(args, "retain_lambda") else 1.0
        )
        self.temperature = args.temperature if hasattr(args, "temperature") else 1.0

        torch.manual_seed(args.seed if hasattr(args, "seed") else 42)

        print(
            f"Initialized FlexibleUnlearner with forget_method={forget_method}, retain_method={retain_method}"
        )

    def compute_forget_loss(self, outputs, labels):
        """Compute the loss for the forget set based on the selected method."""
        if self.forget_method == "GA":
            # Gradient Ascent: Maximize the cross-entropy loss (minimize negative)
            return -self.forget_lambda * self.criterion(outputs, labels)

        elif self.forget_method == "NPO":
            # Negative Preference Optimization: Adapted from DPO for classification
            # Lower the likelihood of correct predictions on forget set
            probs = softmax(outputs, dim=1)
            target_probs = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1)

            # NPO loss: maximize difference between target distribution and model distribution
            # Higher values for incorrect classes, lower for correct class
            npo_loss = (
                -self.forget_lambda
                * torch.sum(target_probs * torch.log(probs + 1e-8), dim=1).mean()
            )
            return npo_loss

        else:
            raise ValueError(f"Unknown forget method: {self.forget_method}")

    def compute_retain_loss(self, outputs, labels, inputs=None):
        """Compute the loss for the retain set based on the selected method."""
        if self.retain_method is None:
            # No retention strategy
            return torch.tensor(0.0, device=self.device)

        elif self.retain_method == "GDR":
            # Gradient Descent on Retain set: Standard cross-entropy loss
            return self.retain_lambda * self.criterion(outputs, labels)

        elif self.retain_method == "KLR":
            # KL Divergence Minimization: Keep predictions similar to original model
            with torch.no_grad():
                original_outputs = self.original_model(inputs)
                original_probs = softmax(original_outputs / self.temperature, dim=1)

            # Compute KL divergence between unlearned and original model predictions
            log_probs = log_softmax(outputs / self.temperature, dim=1)
            kl_loss = self.retain_lambda * kl_div(
                log_probs, original_probs, reduction="batchmean"
            )
            return kl_loss

        else:
            raise ValueError(f"Unknown retain method: {self.retain_method}")

    def train_step(self, images, labels, forget=False):
        """Perform a single training step on a batch."""
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        outputs = self.model(images)

        if forget:
            # Apply the forget set method
            loss = self.compute_forget_loss(outputs, labels)
        else:
            # Apply the retain set method (if any)
            loss = self.compute_retain_loss(outputs, labels, images)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run_unlearning(
        self,
        forget_loader,
        partial_forget_loader,
        retain_loader,
        val_forget_loader,
        val_retain_loader,
        test_loader=None,
    ):
        """Run the unlearning process using the configured methods."""
        print(
            f"Starting {self.forget_method}"
            + (f"_{self.retain_method}" if self.retain_method else "")
            + f" Unlearning for {self.args.unlearn_epochs} epochs..."
        )

        best_model_state = self.model.state_dict()
        # patience = getattr(self.args, "patience", 3)
        patience = 5
        epochs_no_improve = 0
        best_score = float("-inf")

        for epoch in range(1, self.args.unlearn_epochs + 1):
            self.model.train()
            forget_losses = []
            retain_losses = []

            # Process forget set if available
            if partial_forget_loader:
                for forget_images, forget_labels in tqdm(
                    partial_forget_loader, desc=f"Epoch {epoch} (Forget)"
                ):
                    forget_loss = self.train_step(
                        forget_images, forget_labels, forget=True
                    )
                    forget_losses.append(forget_loss)

            # Process retain set if available and a retain method is specified
            if retain_loader and self.retain_method:
                for retain_images, retain_labels in tqdm(
                    retain_loader, desc=f"Epoch {epoch} (Retain)"
                ):
                    retain_loss = self.train_step(
                        retain_images, retain_labels, forget=False
                    )
                    retain_losses.append(retain_loss)

            # Evaluate performance
            self.model.eval()
            forget_metrics = (
                evaluate_model(self.model, forget_loader, self.device)
                if forget_loader
                else {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "loss": 0,
                }
            )
            retain_metrics = (
                evaluate_model(self.model, retain_loader, self.device)
                if retain_loader
                else {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "loss": 0,
                }
            )
            val_forget_metrics = (
                evaluate_model(self.model, val_forget_loader, self.device)
                if val_forget_loader
                else {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "loss": 0,
                }
            )
            val_retain_metrics = (
                evaluate_model(self.model, val_retain_loader, self.device)
                if val_retain_loader
                else {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "loss": 0,
                }
            )

            # Calculate unlearning score (lower accuracy on forget set, higher on retain set)
            forget_acc = forget_metrics["accuracy"] / 100
            retain_acc = retain_metrics["accuracy"] / 100
            val_forget_acc = val_forget_metrics["accuracy"] / 100
            val_retain_acc = val_retain_metrics["accuracy"] / 100
            train_unlearning_score = (1.0 - forget_acc) * retain_acc
            val_unlearning_score = (1.0 - val_forget_acc) * val_retain_acc

            print(f"\nEpoch {epoch} Unlearning Metrics:")
            print(
                f"Train Forget Set  - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}"
            )
            print(
                f"Train Retain Set  - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}"
            )
            print(
                f"Val Forget Set  - Acc: {val_forget_metrics['accuracy']:.2f}%, Loss: {val_forget_metrics['loss']:.4f}"
            )
            print(
                f"Val Retain Set  - Acc: {val_retain_metrics['accuracy']:.2f}%, Loss: {val_retain_metrics['loss']:.4f}"
            )
            print(f"Train Unlearning Score: {train_unlearning_score:.4f}")
            print(f"Val Unlearning Score: {val_unlearning_score:.4f}")
            print("----------------------------------------------------")

            # Early stopping logic
            if val_unlearning_score > best_score:
                best_score = val_unlearning_score
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
            if forget_acc <= 1e-3 or val_forget_acc <= 1e-3:
                break
        # Restore best model
        self.model.load_state_dict(best_model_state)
        return self.model

    def save_model(self, path):
        """Save the unlearned model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Unlearned model saved to {path}")

