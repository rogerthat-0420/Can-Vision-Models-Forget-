import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Can Vision Models Forget")
    
    # Training parameters
    parser.add_argument('--og_batch_size', type=int, default=128, help='Batch size for og training and evaluation')
    parser.add_argument('--og_epochs', type=int, default=200, help='Number of training epochs for og model')
    parser.add_argument('--og_learning_rate', type=float, default=0.1, help='Initial learning rate for og model')
    parser.add_argument('--og_momentum', type=float, default=0.9, help='SGD momentum value for og model')
    parser.add_argument('--og_weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization) for og model')
    parser.add_argument('--og_step_size', type=int, default=60, help='Step size for learning rate scheduler for og model')
    parser.add_argument('--og_gamma', type=float, default=0.2, help='Learning rate decay factor for og model')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Number of epochs to wait before early stopping')
    
    args = parser.parse_args()
    return args