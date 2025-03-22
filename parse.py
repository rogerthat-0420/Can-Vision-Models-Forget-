import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Can Vision Models Forget")

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--train', action='store_true', help="Train the model if set", default=True)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])

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
        default=0.01,
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
    
    parser.add_argument(
        "--gold_standard_class",
        type=int,
        default=None,
        help="Only for getting the gold standard unlearnt model"
    )

    # Unlearning-specific parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/resnet50_cifar10.pth",
        help="Path to the trained model",
    )
    # parser.add_argument(
    #     "--forget_ratio", type=float, default=0.1, help="Ratio of data to forget (0-1)"
    # )
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
        "--output_path",
        type=str,
        default="models/unlearned_model.pth",
        help="Path to save the unlearned model",
    )
    parser.add_argument(
        "--forget_class", type=int, default=0, help="Index of the forget class"
    )

    args = parser.parse_args()
    return args
