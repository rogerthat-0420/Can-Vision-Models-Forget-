import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Can Vision Models Forget")

    # General parameters
    parser.add_argument('--model', type=str, default='resnet50', help='Model name', choices=['resnet50', 'ViT'])
    parser.add_argument('--train_clean', action='store_true', help="Train the model if set")
    parser.add_argument('--train_poisoned', action='store_true', help="Train the poisoned model if set")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'])

    # Unlearing general parameters
    parser.add_argument('--unlearn_mode', type=str, default=None, choices=['class', 'confuse'],
                    help='Unlearning mode: class, confuse (flip A/B), subset')
    parser.add_argument('--forget_class', type=int, default=0, help='Class to unlearn if mode=class')
    parser.add_argument('--class_a', type=int, help='Class A to confuse', default=0)
    parser.add_argument('--class_b', type=int, help='Class B to confuse', default=1)
    parser.add_argument('--df_size', type=float, default=0.5, help='Fraction (0 to 1) controlling number of flipped/confused/subset samples to unlearn')

    # Training parameters
    parser.add_argument(
        "--og_batch_size",
        type=int,
        default=128,
        help="Batch size for og training and evaluation",
    )
    parser.add_argument(
        "--og_epochs",
        type=int,
        default=200,
        help="Number of training epochs for og model",
    )
    parser.add_argument(
        "--og_learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate for og model",
    )
    parser.add_argument(
        "--og_momentum", type=float, default=0.9, help="SGD momentum value for og model"
    )
    parser.add_argument(
        "--og_weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 regularization) for og model",
    )
    parser.add_argument(
        "--og_step_size",
        type=int,
        default=60,
        help="Step size for learning rate scheduler for og model",
    )
    parser.add_argument(
        "--og_gamma",
        type=float,
        default=0.2,
        help="Learning rate decay factor for og model",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs to wait before early stopping",
    )
    
    # parser.add_argument(
    #     "--gold_standard_class",
    #     type=int,
    #     default=None,
    #     help="Only for getting the gold standard unlearnt model"
    # )

    # Unlearning parameters
    parser.add_argument(
        "--unlearn_batch_size", type=int, default=32, help="Batch size for unlearning"
    )
    parser.add_argument(
        "--unlearn_lr", type=float, default=0.0001, help="Learning rate for unlearning"
    )
    parser.add_argument(
        "--unlearn_weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for unlearning optimizer",
    )
    parser.add_argument(
        "--potion_lambda",
        type=float,
        default=0.1,
        help="Lambda parameter for Potion loss",
    )
    parser.add_argument(
        "--unlearn_epochs", type=int, default=100, help="Number of epochs for unlearning"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )  
    parser.add_argument(
        "--skip_training", type=int, default=0, help="Skip initial model training"
    )
    parser.add_argument(
        "--skip_poisoning", type=int, default=0, help="Skip initial model training"
    )  

    #ViT parameters
    parser.add_argument("--embed_dim", type=int, default=128, help='dimensionality of the latent space')
    parser.add_argument("--n_attention_heads", type=int, default=4, help='number of heads to use in Multi-head attention')
    parser.add_argument("--forward_mul", type=int, default=2, help='forward multiplier')
    parser.add_argument("--n_layers", type=int, default=6, help='number of encoder layers')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout value')
    parser.add_argument("--image_size", type=int, default=32, help='image size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument('--vit_lr', type=int, default=5e-4, help='number of epochs to warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument("--n_channels", type=int, default=3, help='number of channels')
    args = parser.parse_args()
    return args
