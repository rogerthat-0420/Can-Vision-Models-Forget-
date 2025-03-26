import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50
from utils import EmbedLayer, Classifier, vit_init_weights


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # initializing the new fc layer properly.
        init.kaiming_normal_(self.model.fc.weight)
        if self.model.fc.bias is not None:
            init.zeros_(self.model.fc.bias)

    def forward(self, x):
        return self.model(x)


class ViT(nn.Module):
    """
    Vision Transformer Class with Pytorch transformer layers (TransformerEncoder and TransformerEncoderLayer) instead of scratch implementation.
    These layer replace the encoder layers (including self-attention operation). Hence, SelfAttention and Encoder classes can be removed.
    Embed layer cannot be replaced as Image to PatchEmbedding Block not available in PyTorch yet.
    Classifier is a simple MLP (and not replaced/not available in PyTorch).

    Parameters:
        n_channels (int)        : Number of channels of the input image
        embed_dim  (int)        : Embedding dimension
        n_layers   (int)        : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        image_size (int)        : Image size
        patch_size (int)        : Patch size
        n_classes (int)         : Number of classes
        dropout  (float)        : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """

    def __init__(
        self,
        n_channels,
        embed_dim,
        n_layers,
        n_attention_heads,
        forward_mul,
        image_size,
        patch_size,
        n_classes,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = EmbedLayer(
            n_channels, embed_dim, image_size, patch_size, dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_attention_heads,
            dim_feedforward=forward_mul * embed_dim,
            dropout=dropout,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)  # Weight initalization

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def get_model(name, num_classes, args):
    if name == "resnet50":
        return ResNet50(num_classes)
    elif name == "vit" or name == "ViT":
        return ViT(
            n_channels=args.n_channels,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_attention_heads=args.n_attention_heads,
            forward_mul=args.forward_mul,
            image_size=args.image_size,
            patch_size=args.patch_size,
            n_classes=num_classes,
            dropout=args.dropout,
        )
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
