import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
import seaborn as sns
import torch.distributed as dist
import torch.nn.functional as F
import copy
import torch



def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor



def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    print("Seed set", seed)

def weight_visualize(weight, modal):
    num_modal = len(modal)
    fig, axes = plt.subplots(1, num_modal, figsize=(12, 4))
    for i, m in enumerate(modal):
        sns.heatmap(weight[m], ax=axes[i], cmap='Blues') 
    return fig
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def info_nce_loss(rep_m: torch.Tensor, rep_a: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss for two batches of representations.

    Args:
        rep_m: torch.Tensor of shape [N, D]
        rep_a: torch.Tensor of shape [N, D]
        temperature: scaling factor for logits

    Returns:
        loss: scalar tensor
    """
    # Normalize to unit sphere (cosine similarity)
    rep_m = F.normalize(rep_m, dim=1)
    rep_a = F.normalize(rep_a, dim=1)

    N = rep_m.size(0)

    # Similarity matrix [N, N]
    logits = torch.matmul(rep_m, rep_a.T) / temperature

    # Targets are diagonal indices (positive pairs)
    labels = torch.arange(N, device=rep_m.device)

    # Cross entropy loss
    loss_m2a = F.cross_entropy(logits, labels)
    loss_a2m = F.cross_entropy(logits.T, labels)

    # Symmetrized loss
    loss = (loss_m2a + loss_a2m) / 2
    return loss

import torch
import matplotlib.pyplot as plt

def compare_tensor_distributions(t1: torch.Tensor, 
                                 t2: torch.Tensor, 
                                 labels=("Tensor 1", "Tensor 2"), 
                                 bins=200,
                                 path="figures/abc"):
    """
    Compare the distributions of three tensors by plotting their histograms.

    Args:
        t1, t2, t3: torch.Tensor
            Input tensors.
        labels: tuple of str
            Labels for each tensor.
        bins: int
            Number of bins for the histogram.
    """
    # Convert to numpy
    arr1 = t1.detach().cpu().numpy().flatten()
    arr2 = t2.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(arr1, bins=bins, alpha=0.5, label=labels[0], density=True)
    plt.hist(arr2, bins=bins, alpha=0.5, label=labels[1], density=True)

    plt.title("Comparison of Tensor Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    # Save plot
    plt.savefig(f"{path}_distribution.png")

def compare_tensor_kde(t1, t2, labels=("Tensor 1", "Tensor 2"), 
                       path="figures/abc", value_range=None):
    """
    Plot KDE distributions of two tensors, optionally restricted to a value range.

    Args:
        t1, t2: torch.Tensor
            Input tensors.
        labels: tuple of str
            Labels for each tensor.
        path: str
            Output file path prefix.
        value_range: tuple (low, high) or None
            If provided, only plot values within [low, high].
    """
    arr1 = t1.detach().cpu().numpy().flatten()
    arr2 = t2.detach().cpu().numpy().flatten()

    # Apply range filtering if specified
    if value_range is not None:
        low, high = value_range
        arr1 = arr1[(arr1 >= low) & (arr1 <= high)]
        arr2 = arr2[(arr2 >= low) & (arr2 <= high)]

    plt.figure(figsize=(8,5))
    sns.kdeplot(arr1, label=labels[0], fill=True, alpha=0.4, clip=value_range)
    sns.kdeplot(arr2, label=labels[1], fill=True, alpha=0.4, clip=value_range)

    plt.title("Tensor Value Distributions (KDE)")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    if value_range is not None:
        plt.xlim(value_range)
    plt.savefig(f"{path}_kde.png", dpi=300, bbox_inches="tight")
    plt.close()

def compare_tensor_scatter(t1, t2, path="figures/abc"):
    arr1 = t1.detach().cpu().numpy().flatten()
    arr2 = t2.detach().cpu().numpy().flatten()

    plt.figure(figsize=(6,6))
    plt.scatter(arr1, arr2, alpha=0.3, s=5)
    plt.xlabel("Tensor 1 values")
    plt.ylabel("Tensor 2 values")
    plt.title("Tensor Value Correlation")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"{path}_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

def compare_tensor_diff(t1, t2, path="figures/abc"):
    diff = (t1 - t2).detach().cpu().numpy()
    plt.figure(figsize=(8,5))
    plt.imshow(diff, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Difference")
    plt.title("Elementwise Difference Heatmap")
    plt.savefig(f"{path}_diff.png", dpi=300, bbox_inches="tight")
    plt.close()

def forward_masked_augmented(model, data_versions, device="cuda"):
    num_versions = len(data_versions)
    num_masked = num_versions - 3   # exclude ori + 2 augmented
    num_augmented = 2

    models = [copy.deepcopy(model).to(device).eval()
              for _ in range(num_versions - 1)]

    rep_masked, rep_augmented = [], []
    with torch.no_grad():
        # masked reps
        for m, data in zip(models[:num_masked], data_versions[1:1+num_masked]):
            _, _, rep = m.net(data)   # << unpack here
            rep_masked.append(rep)

        # augmented reps
        for m, data in zip(models[num_masked:], data_versions[-num_augmented:]):
            _, _, rep = m.net(data)   # << unpack here
            rep_augmented.append(rep)

    del models
    torch.cuda.empty_cache()
    return rep_masked, rep_augmented


####
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a folder for saving plots
os.makedirs("viz_embeddings", exist_ok=True)

def to_2d(x):
    """
    Convert [batch, seq, dim] or [batch, dim] to [batch*seq, dim]
    """
    x = x.detach().cpu()
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    return x.numpy()

def visualize_embeddings(m1, m2, m3, epoch, method="pca", n_samples=200):
    """
    m1, m2, m3: tuples of (hat, hat1, hat2), each [batch, dim]
    method: "pca" or "tsne"
    """

    # Collect embeddings and labels
    all_embeds = []
    all_labels = []

    for i, (hat, hat1, hat2) in enumerate([m1, m2, m3], start=1):
        hat = to_2d(hat)
        hat1 = to_2d(hat1)
        hat2 = to_2d(hat2)

        # sample to avoid clutter
        idx = np.random.choice(hat.shape[0], min(n_samples, hat.shape[0]), replace=False)

        all_embeds.append(hat[idx])
        all_labels.extend([f"U{i}"] * len(idx))
        # all_embeds.append(hat1[idx])
        # all_labels.extend([f"R{i}"] * len(idx))
        # all_embeds.append(hat2[idx])
        # all_labels.extend([f"S{i}"] * len(idx))

    all_embeds = np.concatenate(all_embeds, axis=0)

    # Projection
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, init="pca", random_state=42, perplexity=30)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    proj = reducer.fit_transform(all_embeds)

    # Set global font sizes for paper-ready plots
    plt.rcParams.update({
        "font.size": 18,        # Base font size
        "axes.titlesize": 20,   # Title font
        "axes.labelsize": 18,   # X/Y label font
        "legend.fontsize": 16,  # Legend font
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    })

    # Plot
    plt.figure(figsize=(8, 7))
    for lbl in set(all_labels):
        idx = [i for i, l in enumerate(all_labels) if l == lbl]
        plt.scatter(proj[idx, 0], proj[idx, 1], label=lbl, alpha=0.6, s=30)

    plt.title(f"Embedding projection ({method.upper()}) - Epoch {epoch}")
    plt.legend(markerscale=1.5)  # Make legend markers bigger
    plt.tight_layout()
    plt.savefig(f"viz_embeddings/embeddings_{method}_epoch{epoch}.png", dpi=300)  # dpi=300 for paper
    plt.close()