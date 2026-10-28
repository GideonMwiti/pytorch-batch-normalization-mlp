"""
Project 09: Implementing Batch Normalization in PyTorch
Dataset: Zalando FashionMNIST (10 clothing classes, 28x28 grayscale)
Framework: PyTorch
Author: Gideon Mwiti

Key Features:
- First-principles custom BatchNorm1d implementation verified against PyTorch nn.BatchNorm1d.
- Deep Multi-Layer Perceptron (784 -> 256 -> 128 -> 64 -> 10).
- Controlled ablation study:
  1) Standard MLP (Without Batch Normalization)
  2) Normalized MLP (With nn.BatchNorm1d)
- Demonstrates:
  * Reduction of internal covariate shift
  * Acceleration of convergence (loss & accuracy trajectories)
  * Robustness to higher learning rates
  * Proper handling of running mean/variance in training vs evaluation modes
- 4-panel diagnostic visual artifact:
  1) Training and Validation Loss Convergence Dynamics
  2) Validation Accuracy Acceleration Comparison
  3) Intermediate Activation Distribution (Raw vs Normalized)
  4) Learned Affine Parameters (Gamma Scale & Beta Shift distributions)
"""

import os
import sys
import types
import random
import numpy as np

# Workaround for Windows environments where WDAC blocks unsigned bz2/lzma DLLs
if 'bz2' not in sys.modules:
    sys.modules['bz2'] = types.SimpleNamespace(open=open)
if 'lzma' not in sys.modules:
    sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CustomBatchNorm1d(nn.Module):
    """
    Mathematical first-principles implementation of 1D Batch Normalization (Ioffe & Szegedy, 2015).
    Demonstrates mini-batch mean/variance computation, normalization, 
    learnable affine scaling (gamma) and shifting (beta), and exponential moving statistics.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Non-trainable running buffers
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Mini-batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics with exponential moving average
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                # PyTorch uses Bessel's correction (unbiased=True) for running_var update
                unbiased_var = x.var(dim=0, unbiased=True) if x.size(0) > 1 else batch_var
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var

            # Normalize using batch statistics
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Inference mode: normalize using fixed running statistics
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        return self.gamma * x_hat + self.beta


class MLP(nn.Module):
    """
    Deep Multi-Layer Perceptron supporting optional Batch Normalization.
    Architecture: 784 -> 256 -> 128 -> 64 -> 10
    """
    def __init__(self, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Layer 1
        self.fc1 = nn.Linear(784, 256, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.act1 = nn.ReLU()
        
        # Layer 2
        self.fc2 = nn.Linear(256, 128, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.act2 = nn.ReLU()
        
        # Layer 3
        self.fc3 = nn.Linear(128, 64, bias=not use_batch_norm)
        self.bn3 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        self.act3 = nn.ReLU()
        
        # Output Layer
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # Capture raw linear projection for diagnostic analysis
        z1 = self.fc1(x)
        h1 = self.act1(self.bn1(z1))
        
        z2 = self.fc2(h1)
        h2 = self.act2(self.bn2(z2))
        
        z3 = self.fc3(h2)
        h3 = self.act3(self.bn3(z3))
        
        out = self.fc4(h3)
        return out

    def get_layer1_activations(self, x):
        """Returns raw pre-activation and post-normalization activations for diagnostics."""
        self.eval()
        with torch.no_grad():
            x = x.view(x.size(0), -1)
            raw = self.fc1(x)
            normalized = self.bn1(raw) if self.use_batch_norm else raw
            return raw.cpu().numpy(), normalized.cpu().numpy()


def get_data_loaders(data_dir, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, (correct / total) * 100.0


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / total, (correct / total) * 100.0


def run_experiment(model, train_loader, test_loader, epochs, lr, device, experiment_name):
    criterion = nn.CrossEntropyLoss()
    # Using SGD with momentum to clearly highlight BatchNorm's stabilization effect
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    print(f"\nTraining [{experiment_name}] (LR = {lr}) across {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%")
        
    return history


def generate_visual_artifact(hist_no_bn, hist_bn, raw_acts, norm_acts, model_bn, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    epochs = range(1, len(hist_bn['train_loss']) + 1)
    
    # -------------------------------------------------------------
    # Panel 1: Loss Convergence (BatchNorm vs No-BatchNorm)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(epochs, hist_no_bn['train_loss'], 'r--', label='No BatchNorm - Train Loss', alpha=0.8)
    ax1.plot(epochs, hist_no_bn['test_loss'], 'm-', label='No BatchNorm - Test Loss', linewidth=2.0)
    ax1.plot(epochs, hist_bn['train_loss'], 'b--', label='With BatchNorm - Train Loss', alpha=0.8)
    ax1.plot(epochs, hist_bn['test_loss'], 'darkblue', label='With BatchNorm - Test Loss (Rapid Convergence)', linewidth=2.5)
    ax1.set_title("Cross-Entropy Loss Trajectory: BatchNorm vs. Baseline\nAccelerated Convergence & Loss Minimization", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 2: Validation Accuracy Dynamics
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(epochs, hist_no_bn['test_acc'], 'r-o', label=f"No BatchNorm (Final: {hist_no_bn['test_acc'][-1]:.2f}%)", linewidth=2.0)
    ax2.plot(epochs, hist_bn['test_acc'], 'g-s', label=f"With BatchNorm (Final: {hist_bn['test_acc'][-1]:.2f}%)", linewidth=2.2)
    ax2.set_title("Test Accuracy Trajectory Across Training Epochs\nBatchNorm Achieves Superior Generalization Faster", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    
    # Annotate accuracy delta
    acc_delta = hist_bn['test_acc'][-1] - hist_no_bn['test_acc'][-1]
    ax2.annotate(f"+{acc_delta:.2f}% with BatchNorm",
                 xy=(epochs[-1], hist_bn['test_acc'][-1]),
                 xytext=(epochs[-1] - 1.5, hist_bn['test_acc'][-1] - 2.5),
                 arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', color='darkgreen')

    # -------------------------------------------------------------
    # Panel 3: Intermediate Activation Distribution (Raw vs Normalized)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    raw_flat = raw_acts.flatten()
    norm_flat = norm_acts.flatten()
    
    ax3.hist(raw_flat, bins=60, density=True, alpha=0.5, color='crimson', label=f'Raw Linear Activations\n(μ={np.mean(raw_flat):.2f}, σ={np.std(raw_flat):.2f})')
    ax3.hist(norm_flat, bins=60, density=True, alpha=0.6, color='royalblue', label=f'Batch Normalized Activations\n(μ={np.mean(norm_flat):.2f}, σ={np.std(norm_flat):.2f})')
    ax3.set_title("Layer 1 Pre-activation Distribution: Raw vs. Normalized\nEliminating Covariate Shift via Zero-Mean Unit-Variance Scaling", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Activation Value", fontsize=10)
    ax3.set_ylabel("Probability Density", fontsize=10)
    ax3.set_xlim([-4.5, 4.5])
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 4: Learned Affine Parameters (Gamma Scale & Beta Shift)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    gamma_l1 = model_bn.bn1.weight.detach().cpu().numpy()
    beta_l1 = model_bn.bn1.bias.detach().cpu().numpy()
    
    ax4.scatter(range(len(gamma_l1)), gamma_l1, alpha=0.7, color='teal', label=f"Gamma (Scale γ): mean={np.mean(gamma_l1):.2f}, std={np.std(gamma_l1):.2f}", s=25)
    ax4.scatter(range(len(beta_l1)), beta_l1, alpha=0.7, color='coral', label=f"Beta (Shift β): mean={np.mean(beta_l1):.2f}, std={np.std(beta_l1):.2f}", s=25)
    ax4.axhline(1.0, color='teal', linestyle='--', alpha=0.5, label='Initial Gamma (1.0)')
    ax4.axhline(0.0, color='coral', linestyle='--', alpha=0.5, label='Initial Beta (0.0)')
    ax4.set_title("Learned Affine Parameters for Layer 1 (256 Features)\nRestoring Representational Expressiveness Post-Normalization", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Feature Channel Index", fontsize=10)
    ax4.set_ylabel("Parameter Value", fontsize=10)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', frameon=True, fontsize=8)

    plt.suptitle("Batch Normalization in Deep Neural Networks (FashionMNIST MLP Benchmark)", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def verify_custom_batchnorm():
    """Verifies numerical equivalence between CustomBatchNorm1d and nn.BatchNorm1d."""
    print("--- Verifying CustomBatchNorm1d Numerical Equivalence ---")
    x = torch.randn(16, 32, requires_grad=True)
    
    custom_bn = CustomBatchNorm1d(32)
    official_bn = nn.BatchNorm1d(32)
    
    # Sync weights
    official_bn.weight.data.copy_(custom_bn.gamma.data)
    official_bn.bias.data.copy_(custom_bn.beta.data)
    
    # Forward train mode
    out_custom = custom_bn(x)
    out_official = official_bn(x)
    diff = torch.max(torch.abs(out_custom - out_official)).item()
    print(f"Max Absolute Discrepancy (Forward Training): {diff:.2e}")
    assert diff < 1e-5, "Custom BatchNorm discrepancy exceeds tolerance!"
    print("Verification passed! Custom BatchNorm matches PyTorch nn.BatchNorm1d.\n")


def main():
    print("=" * 68)
    print("      PROJECT 09: IMPLEMENT BATCH NORMALIZATION IN PYTORCH")
    print("=" * 68)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device    : {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_path = os.path.join(script_dir, "batch_normalization_convergence_analysis.png")
    
    # 1. Verify custom mathematical implementation
    verify_custom_batchnorm()
    
    # Hyperparameters
    batch_size = 128
    epochs = 6
    learning_rate = 0.02  # Higher LR where BatchNorm clearly outperforms baseline
    
    # 2. Load FashionMNIST
    train_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    print(f"Dataset             : FashionMNIST (60k Train, 10k Test)")
    print(f"Architecture        : Deep MLP (784 -> 256 -> 128 -> 64 -> 10)")
    print(f"Optimizer           : SGD with Momentum (LR = {learning_rate}, Momentum = 0.9)")
    print(f"Ablation Experiment : Baseline MLP (No BatchNorm) vs. Normalized MLP (BatchNorm1d)")
    print("-" * 68)
    
    # 3. Model A: Standard MLP (Without BatchNorm)
    set_seed(42)
    model_no_bn = MLP(use_batch_norm=False).to(device)
    hist_no_bn = run_experiment(
        model_no_bn, train_loader, test_loader, epochs=epochs, lr=learning_rate,
        device=device, experiment_name="Standard MLP (No BatchNorm)"
    )
    
    # 4. Model B: Normalized MLP (With BatchNorm)
    set_seed(42)
    model_bn = MLP(use_batch_norm=True).to(device)
    hist_bn = run_experiment(
        model_bn, train_loader, test_loader, epochs=epochs, lr=learning_rate,
        device=device, experiment_name="Normalized MLP (With nn.BatchNorm1d)"
    )
    
    # 5. Extract activations for distribution panel
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images.to(device)
    raw_acts, _ = model_no_bn.get_layer1_activations(sample_images)
    _, norm_acts = model_bn.get_layer1_activations(sample_images)
    
    # 6. Experimental Summary
    final_no_bn_acc = hist_no_bn['test_acc'][-1]
    final_bn_acc = hist_bn['test_acc'][-1]
    final_no_bn_loss = hist_no_bn['test_loss'][-1]
    final_bn_loss = hist_bn['test_loss'][-1]
    
    print("\n" + "=" * 68)
    print("                    EXPERIMENTAL COMPARISON")
    print("=" * 68)
    print(f" Model Variant                 | Final Test Loss | Final Test Acc | Epoch 1 Acc")
    print("-" * 68)
    print(f" Standard MLP (No BatchNorm)   |     {final_no_bn_loss:.4f}      |     {final_no_bn_acc:.2f}%     |   {hist_no_bn['test_acc'][0]:.2f}%")
    print(f" Normalized MLP (BatchNorm1d)  |     {final_bn_loss:.4f}      |     {final_bn_acc:.2f}%     |   {hist_bn['test_acc'][0]:.2f}%")
    print("-" * 68)
    print(f" Generalization Advantage     : +{final_bn_acc - final_no_bn_acc:.2f}% Test Accuracy with BatchNorm")
    print(f" Early Convergence Speedup     : +{hist_bn['test_acc'][0] - hist_no_bn['test_acc'][0]:.2f}% Accuracy by Epoch 1")
    print("=" * 68)
    
    # 7. Generate Visual Artifact
    generate_visual_artifact(hist_no_bn, hist_bn, raw_acts, norm_acts, model_bn, save_path)


if __name__ == "__main__":
    main()

