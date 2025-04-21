import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import argparse
import numpy as np

# Assuming you have:
# embeddings: numpy array of shape (num_samples, embedding_dim)
# labels: numpy array of shape (num_samples,) containing class indices (0-99)

def load_data(embeddings_path):
    """Load embeddings and labels from files."""
    embeddings = torch.load(embeddings_path)['embeddings']
    num_embeddings = embeddings.shape[0]
    dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False)
    labels = torch.tensor(dataset.targets)[:num_embeddings]
    return embeddings, labels

def generate_tsne(embeddings, perplexity=30, n_components=2, n_iter=1000, random_state=42):
    """Generate T-SNE embeddings from high-dimensional embeddings."""
    print(f"Running T-SNE on {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state)
    tsne_result = tsne.fit_transform(embeddings)
    print("T-SNE completed.")
    return tsne_result

def plot_tsne(tsne_result, labels, class_names, selected_classes=None, figsize=(12, 10)):
    """
    Plot T-SNE results with colors by class.
    
    Args:
        tsne_result: T-SNE reduced embeddings (n_samples, 2)
        labels: Class labels for each sample
        class_names: List of class names for CIFAR-100
        selected_classes: List of class indices to include in the plot (None means all)
        figsize: Size of the figure
    """
    # Filter for selected classes if specified
    if selected_classes is not None:
        mask = np.isin(labels, selected_classes)
        tsne_result = tsne_result[mask]
        labels = labels[mask]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get a colormap with enough colors
    num_classes = len(np.unique(labels))
    cmap = plt.cm.get_cmap('tab20', max(20, num_classes))
    
    # Create scatter plot
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                           c=labels, cmap=cmap, alpha=0.7, s=30)
    
    # Create legend
    unique_labels = np.unique(labels)
    handles = []
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)[0]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=cmap(label / max(20, num_classes)), 
                            markersize=8, label=class_names[label]))
    
    plt.legend(handles=handles, loc='best', title="Classes", 
               bbox_to_anchor=(1.05, 1), fontsize='small')
    
    plt.title('T-SNE Visualization of CIFAR-100 Embeddings')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.tight_layout()
    
    return plt

def main(embeddings_path: str, plot_path: str):
    # CIFAR-100 class names
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    # Replace with your embeddings and labels paths

    embeddings, labels = load_data(embeddings_path)
    embeddings = embeddings.mean(dim=1)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    # Generate T-SNE
    tsne_result = generate_tsne(embeddings)
    
    # Example 1: Plot all classes
    selected_classes = list(np.arange(50, 60))
    plt = plot_tsne(tsne_result, labels, class_names, selected_classes)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("T-SNE visualizations completed and saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", dest="embeddings_path")
    parser.add_argument("--plot-path", dest="plot_path")

    args = parser.parse_args()

    main(args.embeddings_path, args.plot_path)
