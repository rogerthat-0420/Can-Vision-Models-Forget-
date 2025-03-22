import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from evaluate import evaluate_model  # Importing your global evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PotionUnlearner:
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.unlearn_lr, weight_decay=args.unlearn_weight_decay)
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
        patience, epochs_no_improve = 3, 0
        best_score = float('-inf')

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

            forget_metrics = evaluate_model(self.model, forget_loader)
            retain_metrics = evaluate_model(self.model, retain_loader)

            forget_acc = forget_metrics['accuracy'] / 100
            retain_acc = retain_metrics['accuracy'] / 100
            unlearning_score = (1.0 - forget_acc) * retain_acc

            print(f"\nEpoch {epoch} Unlearning Metrics:")
            print(f"Forget Set  - Acc: {forget_metrics['accuracy']:.2f}%, Loss: {forget_metrics['loss']:.4f}")
            print(f"Retain Set  - Acc: {retain_metrics['accuracy']:.2f}%, Loss: {retain_metrics['loss']:.4f}")
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
        print(f"✅ Unlearned model saved to {path}")
