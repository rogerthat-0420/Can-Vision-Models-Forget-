import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Union, Optional
import math
import io
import pathlib
from torchvision.utils import make_grid
from matplotlib.colors import LinearSegmentedColormap

from modelopt.torch.quantization.utils import export_torch_mode
import modelopt.torch.opt as mto

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import utils
import models
from quantization_vit import get_quantized_model

class FeatureMapExtractor:
    """Extract feature maps from ResNet and other CNN models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.feature_maps = {}
        self.layers_of_interest = set()

    def register_hooks(self, layers_of_interest: Optional[List[str]] = None) -> None:
        """Register forward hooks on layers of interest.

        Args:
            layers_of_interest: List of layer names to extract features from.
                               If None, will try to automatically select interesting layers.
        """
        # Clear any existing hooks
        self.remove_hooks()
        self.feature_maps = {}

        if layers_of_interest is None:
            # Auto-detect interesting layers for ResNet
            if hasattr(self.model, 'layer1'):  # ResNet structure
                self.layers_of_interest = {
                    'conv1', 'layer1', 'layer2', 'layer3', 'layer4'
                }
            else:
                # For other models, try to find conv layers
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.MaxPool2d):
                        self.layers_of_interest.add(name)
        else:
            self.layers_of_interest = set(layers_of_interest)

        # Register hooks for all modules
        for name, module in self.model.named_modules():
            if name in self.layers_of_interest:
                self.hooks.append(
                    module.register_forward_hook(self._get_hook_fn(name))
                )

    def _get_hook_fn(self, name: str):
        def hook_fn(module, input, output):
            self.feature_maps["embeddings"] = output.detach()
        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and extract features.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Dictionary of feature maps by layer name
        """
        was_training = self.model.training
        self.model.eval()

        self.feature_maps = {}
        with torch.no_grad():
            self.model(x)

        if was_training:
            self.model.train()

        return self.feature_maps

    def __del__(self):
        self.remove_hooks()

class AttentionMapExtractor:
    """Extract attention maps from Vision Transformer models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.attention_maps = {}

    def extract_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and extract attention maps.

        Args:
            x: Input tensor

        Returns:
            Dictionary of attention maps by layer name
        """
        was_training = self.model.training
        self.model.eval()

        self.attention_maps = {}
        with torch.no_grad():
            logits, hidden_states = self.model(x, output_hidden_states=True)

        if was_training:
            self.model.train()

        self.attention_maps = {
            "embeddings": hidden_states[-1]
        }

        return self.attention_maps

    def compute_attention_rollout(self, discard_ratio: float = 0.9) -> np.ndarray:
        """Compute attention rollout from extracted attention maps.

        Args:
            discard_ratio: Ratio of lowest attention values to discard

        Returns:
            Attention rollout map as numpy array
        """
        if not self.attention_maps:
            raise ValueError("No attention maps extracted. Run extract_attention first.")

        # Sort layers by index for proper rollout order
        layer_names = sorted(self.attention_maps.keys(),
                            key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        # Process each layer's attention
        processed_attentions = []
        for name in layer_names:
            attn = self.attention_maps[name]
            # Average over attention heads if needed
            if len(attn.shape) == 4:  # [batch_size, num_heads, seq_len, seq_len]
                attn = attn.mean(dim=1)  # Average over heads

            # Discard low attention values
            if discard_ratio > 0:
                flat_attn = attn.view(attn.size(0), -1)
                k = int((1 - discard_ratio) * flat_attn.size(-1))
                val, idx = flat_attn.topk(k, dim=-1)
                flat_attn = torch.zeros_like(flat_attn).scatter_(-1, idx, val)
                attn = flat_attn.view(attn.shape)

            # Add residual connection as in attention rollout paper
            attn = attn + torch.eye(attn.size(-1), device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            processed_attentions.append(attn)

        # Compute rollout
        rollout = processed_attentions[0]
        for attn in processed_attentions[1:]:
            rollout = torch.matmul(attn, rollout)

        # Extract cls token attention if available (first token)
        rollout_map = rollout[0, 0]  # First batch, first query token

        return rollout_map.cpu().numpy()

class VisualizationTool:
    """Tool for visualizing and comparing ResNet feature maps and ViT attention maps."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_extractor = None
        self.vit_extractor = None
        self.last_input = None
        self.last_processed_image = None

        # Colormaps for visualizations
        self.feature_cmap = plt.cm.viridis
        self.attention_cmap = plt.cm.inferno

    def set_resnet_model(self, model: Optional[nn.Module] = None, pretrained: bool = True, quantized: bool = False) -> None:
        """Set ResNet model for feature extraction.

        Args:
            model: Pre-instantiated model or None to use default ResNet50
            pretrained: Whether to use pretrained weights (when model is None)
        """
        if model is None:
            model = models.resnet50(pretrained=pretrained).to(self.device)
        else:
            model = model.to(self.device)

        self.resnet_extractor = FeatureMapExtractor(model)
        self.resnet_extractor.register_hooks()
        self.quantized = quantized

    def set_vit_model(self, model: nn.Module) -> None:
        """Set ViT model for attention map extraction.

        Args:
            model: Vision Transformer model
        """
        model = model.to(self.device)
        self.vit_extractor = AttentionMapExtractor(model)

    def extract_resnet_features(self, x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract ResNet feature maps.

        Args:
            x: Input tensor or None to use last input

        Returns:
            Dictionary of feature maps
        """
        if self.resnet_extractor is None:
            raise ValueError("ResNet extractor not initialized. Call set_resnet_model first.")

        if x is None:
            if self.last_input is None:
                raise ValueError("No input tensor available.")
            x = self.last_input
            
        if self.quantized:
            with export_torch_mode():
                return self.resnet_extractor.extract_features(x)
        else:
            return self.resnet_extractor.extract_features(x)

    def extract_vit_features(self, x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract ViT attention maps.

        Args:
            x: Input tensor or None to use last input

        Returns:
            Dictionary of attention maps
        """
        if self.vit_extractor is None:
            raise ValueError("ViT extractor not initialized. Call set_vit_model first.")

        if x is None:
            if self.last_input is None:
                raise ValueError("No input tensor available.")
            x = self.last_input
        return self.vit_extractor.extract_attention(x)

    def _overlay_attention_on_image(self, attention_map: np.ndarray) -> None:
        """Overlay attention map on original image.

        Args:
            attention_map: Attention map as numpy array
        """
        if self.last_processed_image is None:
            print("No original image available for overlay")
            return

        # Resize attention map to match image size
        h, w = self.last_processed_image.size
        resized_attn = np.array(Image.fromarray(attention_map).resize((w, h)))

        # Normalize for visualization
        attention_map_norm = (resized_attn - resized_attn.min()) / (resized_attn.max() - resized_attn.min() + 1e-8)

        # Create heatmap overlay
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(self.last_processed_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(attention_map_norm, cmap=self.attention_cmap)
        plt.title("Attention Map")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(self.last_processed_image)
        plt.imshow(attention_map_norm, cmap=self.attention_cmap, alpha=0.7)
        plt.title("Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize and compare ResNet and ViT')

    # Image loading
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", dest="og_batch_size", type=int, default=8)

    # quantization args
    parser.add_argument("--quantized", dest="quantized", action='store_true')
    parser.add_argument("--quantization-type", dest='quantization_type', default=None)

    # Model selection
    parser.add_argument('--resnet_model', type=pathlib.Path,
                        help='ResNet model variant')
    parser.add_argument('--vit_model', type=pathlib.Path,
                        help='Path to ViT model checkpoint')

    # Visualization options
    parser.add_argument('--resnet_layer', type=str,
                        help='ResNet layer to visualize (e.g., layer1, layer2)')
    parser.add_argument('--vit_layer', type=int,
                        help='ViT layer index to visualize')
    parser.add_argument('--vit_head', type=int,
                        help='ViT attention head to visualize')
    parser.add_argument('--top-k', dest='top_k', type=int, default=16,
                        help='Number of top feature channels to show')

    # Mode
    parser.add_argument('--mode', type=str,
                        choices=['resnet50', 'vit', 'rollout'],
                        help='Visualization mode')
    parser.add_argument('--plot-path', type=pathlib.Path, dest='output_path', help="where to store the output plot?")
    parser.add_argument('--tensor-path', type=pathlib.Path, dest='tensor_path', help="where to store the tensors?")

    return parser.parse_args()

def main():
    """Main function for command-line usage."""
    args = parse_arguments()
    args.model = args.mode
    args.dataset = 'cifar100'
    # Create tool
    tool = VisualizationTool()

    # Load image
    _, _, test_dataloader, num_classes, _, _, test_dataset = utils.load_dataset(args, device)

    # Load models
    resnet_model = None
    if args.mode in ['resnet50']:
        # ensure we load correct model
        resnet_model = models.get_model(args.mode, num_classes)

        if args.quantized:
            mto.restore(resnet_model, args.resnet_model)
            resnet_model = torch.compile(resnet_model, backend="inductor")
        else:
            resnet_model.load_state_dict(torch.load(args.resnet_model))

        resnet_model = resnet_model.eval().to(device)
        tool.set_resnet_model(resnet_model, quantized=args.quantized)


    if args.mode in ['vit']:
        # Load custom ViT model
        # ensure that we load correct model
        vit_model = models.get_model(args.mode, num_classes)
        vit_model.load_state_dict(torch.load(args.vit_model))

        if args.quantized:
            vit_model = get_quantized_model(vit_model, args.vit_model, args.quantization_type)

        vit_model.eval()
        tool.set_vit_model(vit_model)

    # Perform visualization based on mode
    features = []
    for idx, (img, label) in enumerate(test_dataloader):
        if idx > 125:
            break
        img = img.to(device)
        if args.mode == 'resnet50':
            features.append(tool.extract_resnet_features(img))
        elif args.mode == 'vit':
            features.append(tool.extract_vit_features(img))

    final_features = {
        k: () for k in features[0]
    }

    for f in features:
        for k, v in f.items():
            final_features[k] = final_features[k] + (v.cpu(),)

    final_features = {
        k: torch.cat(v) for k, v in final_features.items()
    }

    torch.save(final_features, args.tensor_path)
    print(f"saving in {args.tensor_path}")
        

# Example of usage in a notebook or script
if __name__ == "__main__":
    main()
