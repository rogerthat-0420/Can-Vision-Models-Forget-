import torch
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(0)

# Dummy predictions from unlearned model (logits)
logits = torch.randn(5, 10)  # 5 samples, 10 classes

# Convert logits to log-probabilities
temperature = 1.0
log_probs = F.log_softmax(logits / temperature, dim=1)

# Dummy predictions from original model (logits), treated as "target"
original_logits = torch.randn(5, 10)
original_probs = F.softmax(original_logits / temperature, dim=1)

# KL divergence
kl_loss = F.kl_div(log_probs, original_probs, reduction="batchmean")

# If using a scaling factor (retain_lambda), apply it here
retain_lambda = 1.0
kl_loss = retain_lambda * kl_loss

print("KL Divergence Loss:", kl_loss.item())
