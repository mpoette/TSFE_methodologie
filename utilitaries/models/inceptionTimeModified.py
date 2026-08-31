"""
InceptionTime for binary classification of time series
================================================================

Complete and self-contained module integrating:
- InceptionTime architecture (Fawaz et al.)
- Training with early stopping, class weighting, temperature calibration
- Shallow fine-tuning (partial layer freezing)
- Robust evaluation with calibrated metrics

Usage:
    from inception_time import train_inception_time, fine_tune_inception_time, evaluate_on_test

    # Training from scratch
    model, T, history, splits = train_inception_time(X_train, y_train)
    
    # Fine-tuning
    model_ft, T_ft, hist_ft, splits_ft = fine_tune_inception_time(
        "best_model.pt", X_new, y_new, last_k_blocks=2
    )
    
    # Evaluation
    auc, brier, T = evaluate_on_test(X_test, y_test, "best_model.pt")

Author: Adaptation with corrections and improvements
License: MIT
"""

from __future__ import annotations
import math
import os
from typing import Dict, Optional, Tuple, Union, Literal
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm not available - no progress bar")

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available - evaluation metrics unavailable")


# ============================================================================
# ARCHITECTURE INCEPTIONTIME
# ============================================================================

class Conv1dSamePadding(nn.Conv1d):
    """Conv1D with TensorFlow-like 'same' padding."""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride,
            self.dilation, self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    """Implementation of 'same' padding."""
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])
    return F.conv1d(
        input=input, weight=weight, bias=bias, stride=stride,
        padding=padding // 2, dilation=dilation, groups=groups
    )


class InceptionBlock(nn.Module):
    """Inception block with multi-scale convolutions and a residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool,
        stride: int = 1,
        bottleneck_channels: int = 32,
        kernel_size: int = 41,
        num_groups: int = 8
    ) -> None:
        super().__init__()
        assert kernel_size > 3, "kernel_size must be > 3"
        
        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(
                in_channels, bottleneck_channels,
                kernel_size=1, bias=False
            )
        
        # Three kernel scales: k, k/2, k/4
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size_s[i],
                stride=stride,
                bias=False
            )
            for i in range(len(kernel_size_s))
        ])
        
        # GroupNorm for stability with small batches
        self.groupnorm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=channels[-1]
        )
        
        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)
        x = self.groupnorm(x)
        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class InceptionModel(nn.Module):
    """Full InceptionTime model for binary classification."""
    
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: Union[int, list[int]],
        bottleneck_channels: Union[int, list[int]],
        kernel_sizes: Union[int, list[int]],
        use_residuals: Union[bool, list[bool], str] = 'default',
        num_pred_classes: int = 1
    ) -> None:
        super().__init__()
        
        # Save arguments for reconstruction
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes,
        }
        
        # Parameter expansion
        channels = [in_channels] + self._expand_to_blocks(out_channels, num_blocks)
        bottleneck_channels = self._expand_to_blocks(bottleneck_channels, num_blocks)
        kernel_sizes = self._expand_to_blocks(kernel_sizes, num_blocks)
        
        if use_residuals == 'default':
            # Residual connection every 3 blocks (standard InceptionTime)
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = self._expand_to_blocks(use_residuals, num_blocks)
        
        # Building the blocks
        self.blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                residual=bool(use_residuals[i]),
                bottleneck_channels=int(bottleneck_channels[i]),
                kernel_size=int(kernel_sizes[i])
            )
            for i in range(num_blocks)
        ])
        
        # Classification head
        self.linear = nn.Linear(
            in_features=channels[-1],
            out_features=num_pred_classes
        )
    
    @staticmethod
    def _expand_to_blocks(value, num_blocks: int):
        """Converts a scalar or list into a list of length num_blocks."""
        if isinstance(value, list):
            assert len(value) == num_blocks, f"Liste doit avoir {num_blocks} éléments"
            return value
        return [value] * num_blocks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global Average Pooling over the time dimension
        x = self.blocks(x).mean(dim=-1)
        return self.linear(x)


# ============================================================================
# TEMPERATURE CALIBRATION
# ============================================================================

class TemperatureCalibrator(nn.Module):
    """
    Affine calibrator for binary classification.

    The transformation applied to the logits is::

        calibrated_logits = logits / T + bias

    ``T`` corrects the confidence of the model, while ``bias`` corrects the
    global offset of the logits, in particular the one that can be introduced
    by ``pos_weight`` during training.
    """

    def __init__(self, init_T: float = 1.0, init_bias: float = 0.0) -> None:
        super().__init__()
        if init_T <= 0:
            raise ValueError("init_T must be strictly positive")

        self.log_T = nn.Parameter(
            torch.tensor(math.log(init_T), dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor(init_bias, dtype=torch.float32)
        )

    @property
    def T(self) -> torch.Tensor:
        """Returns a strictly positive temperature."""
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies the affine calibration to the raw logits."""
        return logits / self.T + self.bias

    def fit(
        self,
        logits_val: torch.Tensor,
        y_val: torch.Tensor,
        max_iter: int = 200
    ) -> float:
        """
        Learns ``T`` and ``bias`` on the validation set.

        The logits are detached from the main model graph and the two
        calibration parameters are optimized with LBFGS by minimizing the
        unweighted BCE.

        Returns:
            Final value of the calibrated BCE.
        """
        logits_val = logits_val.detach()
        y_val = y_val.detach().float().reshape(-1)
        self.train()

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            calibrated_logits = self(logits_val).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(
                calibrated_logits,
                y_val,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

        with torch.no_grad():
            final_loss = F.binary_cross_entropy_with_logits(
                self(logits_val).reshape(-1),
                y_val,
            )

        return float(final_loss.cpu())


# ============================================================================
# DATASET ET UTILS
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series (N, T, F) -> (N, F, T)."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, "X must be (N, T, F)"
        if y.ndim == 2:
            y = y.reshape(-1)
        
        # Permutation for Conv1D: (N, T, F) -> (N, F, T)
        self.X = torch.from_numpy(X).permute(0, 2, 1).contiguous().float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def stratified_train_val_indices(
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Manual stratified split for binary classification."""
    y = y.reshape(-1)
    rng = np.random.default_rng(seed)
    
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    
    n0_val = int(round(val_ratio * len(idx0)))
    n1_val = int(round(val_ratio * len(idx1)))
    
    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    
    return train_idx, val_idx


def compute_pos_weight_from_indices(
    y: np.ndarray,
    train_idx: np.ndarray
) -> Optional[torch.Tensor]:
    """Computes pos_weight for BCEWithLogitsLoss (class balancing)."""
    ytr = y.reshape(-1)[train_idx]
    
    if set(np.unique(ytr)).issubset({0, 1}):
        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        if pos > 0:
            return torch.tensor([neg / pos], dtype=torch.float32)
    
    return None


def _gather_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """Collects the logits over a DataLoader."""
    was_training = model.training
    model.eval()
    outs = []
    
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            z = model(xb)
            if z.ndim == 1:
                z = z.unsqueeze(-1)
            outs.append(z)
    
    logits = torch.cat(outs, dim=0)
    
    if was_training:
        model.train()
    
    return logits


# ============================================================================
# ENTRAÎNEMENT FROM SCRATCH
# ============================================================================

def train_inception_time(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float = 0.2,
    num_blocks: int = 6,
    out_channels: int = 64,
    bottleneck_channels: int = 32,
    kernel_sizes: int = 41,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    clip_grad: Optional[float] = 1.0,
    use_scheduler: bool = True,
    calibrate: bool = True,
    save_best_path: Optional[str] = "best_inception_time.pt",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    progress: bool = True,
    seed : int = 42,
    X_val: Optional[np.ndarray] = None,  # <-- ADDED: optional parameter
    y_val: Optional[np.ndarray] = None,  # <-- ADDED: optional parameter
) -> Tuple[InceptionModel, float, Dict[str, list], Dict[str, np.ndarray]]:
    """
    Trains InceptionTime from scratch with all the improvements.
    
    Args:
        X: Data (N, T, F)
        y: Binary labels (N,) or (N, 1)
        val_ratio: Validation proportion
        num_blocks: Number of Inception blocks
        out_channels: Output channels per block
        bottleneck_channels: Bottleneck channels
        kernel_sizes: Base kernel size
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        lr: Initial learning rate
        weight_decay: L2 regularization
        clip_grad: Gradient clipping (None to disable)
        use_scheduler: Use CosineAnnealingLR
        calibrate: Calibrate temperature on the validation set
        save_best_path: Save path (None to disable)
        device: PyTorch device
        progress: Show progress bar
    
    Returns:
        (model, T, history, splits)
        - model: Trained model
        - T: Calibrated temperature
        - history: Loss history
        - splits: train/val indices
    """
    # Validations
    assert X.shape[0] == y.reshape(-1).shape[0], "N mismatch between X and y"
    assert X.shape[2] > 0, "X must have at least 1 feature"
    assert X.ndim == 3, f"X doit être (N,T,F), reçu shape {X.shape}"
    
    device = torch.device(device)
    # --- STRICT PYTORCH DETERMINISM CONFIGURATION ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Deterministic generator for the DataLoader
    gen = torch.Generator()
    gen.manual_seed(seed)
    if X_val is not None and y_val is not None:
        assert X_val.ndim == 3, f"X_val doit être (N,T,F), reçu shape {X_val.shape}"
        assert X_val.shape[0] == y_val.reshape(-1).shape[0], "N mismatch between X_val and y_val"
        assert X_val.shape[2] == X.shape[2], "Feature mismatch between X and X_val"
        
        # We use the sets passed as parameters directly
        train_ds = TimeSeriesDataset(X, y)
        val_ds = TimeSeriesDataset(X_val, y_val)
        
        # Default filling to avoid breaking the returned splits dictionaries
        train_idx = np.arange(len(X))
        val_idx = np.arange(len(X_val))
    else:
        # Previous behavior: automatic stratified split
        train_idx, val_idx = stratified_train_val_indices(y, val_ratio=val_ratio, seed=seed)
        ds = TimeSeriesDataset(X, y)
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
    
    # DataLoaders
    ds = TimeSeriesDataset(X, y)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        generator=gen
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )
    
    # Model
    in_channels = X.shape[2]
    model = InceptionModel(
        num_blocks=num_blocks,
        in_channels=in_channels,
        out_channels=out_channels,
        bottleneck_channels=bottleneck_channels,
        kernel_sizes=kernel_sizes,
        use_residuals='default',
        num_pred_classes=1,
    ).to(device)
    
    # Loss with class weighting
    pos_w = compute_pos_weight_from_indices(y, train_idx)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device) if pos_w is not None else None
    )
    
    # Optimiseur
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler (cosine annealing)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    # Early stopping
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}
    patience_count = 0
    
    # Boucle d'entraînement
    if progress and TQDM_AVAILABLE:
        pbar = tqdm(range(1, epochs + 1), desc="Training", leave=True)
    else:
        pbar = range(1, epochs + 1)
    
    for epoch in pbar:
        # ========== TRAIN ==========
        model.train()
        run_loss = 0.0
        n_obs = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)
            
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss non finie à epoch {epoch}")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad
                )
            
            optimizer.step()
            
            bs = xb.size(0)
            run_loss += float(loss.detach().cpu()) * bs
            n_obs += bs
        
        train_loss = run_loss / max(1, n_obs)
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        m_obs = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device)
                
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                
                bs = xb.size(0)
                val_loss += float(loss.detach().cpu()) * bs
                m_obs += bs
        
        val_loss /= max(1, m_obs)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Affichage
        if progress and TQDM_AVAILABLE:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                lr=f"{current_lr:.2e}"
            )
        
        # ========== EARLY STOPPING ==========
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            
            # Sauvegarde checkpoint
            if save_best_path is not None:
                torch.save({
                    'state_dict': best_state,
                    'init_args': model.input_args,
                    'epoch': epoch,
                    'history': history,
                    'train_idx': train_idx,
                    'val_idx': val_idx
                }, save_best_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                if progress and TQDM_AVAILABLE:
                    pbar.set_postfix_str(f"Early stop @ epoch {epoch}")
                break
    
    # Reload best state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Meilleur modèle chargé (epoch {best_epoch}, val_loss={best_val:.6f})")
    
    # ========== TEMPERATURE CALIBRATION ==========
    T_value = 1.0
    calibration_bias = 0.0
    model.temperature_ = T_value
    model.calibration_bias_ = calibration_bias
    if calibrate:
        print("Temperature calibration on the validation set...")
        logits_val = _gather_logits(model, val_loader, device)
        if X_val is not None and y_val is not None:
            y_val = torch.from_numpy(y_val.reshape(-1)).float().to(device)
        else:
            y_val = torch.from_numpy(y.reshape(-1)[val_idx]).to(device)
        
        if logits_val.ndim == 1:
            logits_val = logits_val.unsqueeze(-1)
        
        calibrator = TemperatureCalibrator(init_T=1.0).to(device)
        _ = calibrator.fit(logits_val, y_val, max_iter=200)
        T_value = float(calibrator.T.detach().cpu())
        calibration_bias = float(calibrator.bias.detach().cpu())
        model.temperature_ = T_value
        model.calibration_bias_ = calibration_bias
        
        print(f"✓ Calibration : T = {T_value:.4f}, bias = {calibration_bias:.4f}")
        
        # Update checkpoint with T
        if save_best_path is not None and os.path.exists(save_best_path):
            ckpt = torch.load(save_best_path, map_location='cpu', weights_only=False)
            ckpt['temperature'] = T_value
            ckpt['calibration_bias'] = calibration_bias
            torch.save(ckpt, save_best_path)
    
    # Free GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    splits = {"train_idx": train_idx, "val_idx": val_idx}
    return model, T_value, history, splits


# ============================================================================
# FINE-TUNING SUPERFICIEL
# ============================================================================

def set_trainable_last_k_blocks(
    model: InceptionModel,
    last_k_blocks: int = 1,
    train_linear: bool = True
):
    """
    Freezes all blocks except the last K.
    Uses hooks to keep eval() on the frozen blocks.
    """
    assert isinstance(model.blocks, nn.Sequential), "model.blocks must be nn.Sequential"
    num_blocks = len(model.blocks)
    k = max(0, min(last_k_blocks, num_blocks))
    
    # Freeze everything by default
    for p in model.parameters():
        p.requires_grad = False
    
    # Hook to force eval() even after model.train()
    def force_eval_mode(module, input):
        module.eval()
    
    # Frozen blocks: permanent eval() via hook
    for i in range(0, num_blocks - k):
        blk = model.blocks[i]
        blk.eval()
        blk.register_forward_pre_hook(force_eval_mode)
    
    # Blocs entraînables
    for i in range(num_blocks - k, num_blocks):
        blk = model.blocks[i]
        blk.train()
        for p in blk.parameters():
            p.requires_grad = True
    
    # Linear layer
    if train_linear:
        model.linear.train()
        for p in model.linear.parameters():
            p.requires_grad = True
    else:
        model.linear.eval()
    
    return model


def load_model_from_checkpoint(
    ckpt_path: str,
    device: Union[str, torch.device] = None
) -> Tuple[InceptionModel, dict, Optional[float], float]:
    """Loads the model and its calibration parameters from a checkpoint."""
    device = torch.device(device) if device is not None else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trouvé : {ckpt_path}")
    
    # PyTorch 2.6+: weights_only=False for numpy compatibility
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    init_args = ckpt['init_args']
    state_dict = ckpt.get('state_dict') or ckpt.get('model_state_dict')
    T = ckpt.get('temperature', None)
    calibration_bias = float(ckpt.get('calibration_bias', 0.0))
    
    model = InceptionModel(**init_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, init_args, T, calibration_bias


def fine_tune_inception_time(
    model_or_ckpt: Union[str, InceptionModel],
    X_ft: np.ndarray,
    y_ft: np.ndarray,
    *,
    last_k_blocks: int = 1,
    train_linear: bool = True,
    reinit_linear: bool = False,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
    min_delta: float = 0.0,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    clip_grad: Optional[float] = 0.5,
    use_scheduler: bool = True,
    calibrate: bool = True,
    save_best_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    progress: bool = True,
    seed : int = 42,
) -> Tuple[InceptionModel, float, Dict[str, list], Dict[str, np.ndarray]]:
    """
    Shallow fine-tuning: freeze deep blocks, train the last ones.
    
    Args:
        model_or_ckpt: Checkpoint path OR pre-trained model
        X_ft: New data (N, T, F)
        y_ft: New labels
        last_k_blocks: Number of blocks to unfreeze (from the end)
        train_linear: Train the linear layer
        reinit_linear: Re-initialize the linear layer before FT
        (other args similar to train_inception_time)
    
    Returns:
        (model, T, history, splits)
    """
    device = torch.device(device)

    # --- STRICT PYTORCH DETERMINISM CONFIGURATION ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # Model loading
    if isinstance(model_or_ckpt, str):
        model, init_args, _T, _calibration_bias = load_model_from_checkpoint(model_or_ckpt, device)
        print(f"✓ Modèle chargé depuis {model_or_ckpt}")
    else:
        model = model_or_ckpt.to(device)
        init_args = getattr(model, 'input_args', None) or {}
    
    # Validation shape
    assert X_ft.ndim == 3, f"X_ft doit être (N,T,F), reçu {X_ft.shape}"
    F_in = X_ft.shape[2]
    if init_args and 'in_channels' in init_args:
        F_expected = init_args['in_channels']
        assert F_in == F_expected, f"Mismatch features : modèle attend {F_expected}, X_ft a {F_in}"
    
    # Optional re-initialization of the head
    if reinit_linear:
        print("Re-initializing the linear layer...")
        if hasattr(model.linear, 'reset_parameters'):
            model.linear.reset_parameters()
        else:
            nn.init.kaiming_uniform_(model.linear.weight, a=math.sqrt(5))
            if model.linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.linear.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(model.linear.bias, -bound, bound)
    
    # Configuration fine-tuning
    set_trainable_last_k_blocks(
        model,
        last_k_blocks=last_k_blocks,
        train_linear=train_linear
    )
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"✓ Fine-tuning : {n_trainable:,} / {n_total:,} paramètres entraînables ({100*n_trainable/n_total:.1f}%)")
    
    # Dataset/Loaders
    ds = TimeSeriesDataset(X_ft, y_ft)
    train_idx, val_idx = stratified_train_val_indices(y_ft, val_ratio=val_ratio, seed=seed)
    
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        generator = gen
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )
    
    # Loss with class weight
    pos_w = compute_pos_weight_from_indices(y_ft, train_idx)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device) if pos_w is not None else None
    )
    
    # Optimizer (only on trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    # Early stopping
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}
    patience_count = 0
    
    # Boucle FT
    if progress and TQDM_AVAILABLE:
        pbar = tqdm(range(1, epochs + 1), desc="Fine-tuning", leave=True)
    else:
        pbar = range(1, epochs + 1)
    
    for epoch in pbar:
        # ========== TRAIN ==========
        model.train()  # The frozen blocks stay in eval() thanks to the hooks
        run_loss = 0.0
        n_obs = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)
            
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss non finie à epoch {epoch}")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=clip_grad
                )
            
            optimizer.step()
            
            bs = xb.size(0)
            run_loss += float(loss.detach().cpu()) * bs
            n_obs += bs
        
        train_loss = run_loss / max(1, n_obs)
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        m_obs = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device)
                
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                
                bs = xb.size(0)
                val_loss += float(loss.detach().cpu()) * bs
                m_obs += bs
        
        val_loss /= max(1, m_obs)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Affichage
        if progress and TQDM_AVAILABLE:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                lr=f"{current_lr:.2e}"
            )
        
        # ========== EARLY STOPPING ==========
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            
            if save_best_path is not None:
                torch.save({
                    'state_dict': best_state,
                    'init_args': getattr(model, 'input_args', init_args),
                    'epoch': epoch,
                    'history': history,
                    'train_idx': train_idx,
                    'val_idx': val_idx
                }, save_best_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                if progress and TQDM_AVAILABLE:
                    pbar.set_postfix_str(f"Early stop @ epoch {epoch}")
                break
    
    # Reload best state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Meilleur modèle chargé (epoch {best_epoch}, val_loss={best_val:.6f})")
    
    # ========== CALIBRATION ==========
    T_value = 1.0
    calibration_bias = 0.0
    model.temperature_ = T_value
    model.calibration_bias_ = calibration_bias
    if calibrate:
        print("Temperature calibration on the fine-tuning validation set")
        logits_val = _gather_logits(model, val_loader, device)
        y_val = torch.from_numpy(y_ft.reshape(-1)[val_idx]).to(device)
        
        if logits_val.ndim == 1:
            logits_val = logits_val.unsqueeze(-1)
        
        calibrator = TemperatureCalibrator(init_T=1.0).to(device)
        _ = calibrator.fit(logits_val, y_val, max_iter=200)
        T_value = float(calibrator.T.detach().cpu())
        calibration_bias = float(calibrator.bias.detach().cpu())
        model.temperature_ = T_value
        model.calibration_bias_ = calibration_bias
        
        print(f"✓ Calibration : T = {T_value:.4f}, bias = {calibration_bias:.4f}")
        
        if save_best_path is not None and os.path.exists(save_best_path):
            ckpt = torch.load(save_best_path, map_location='cpu', weights_only=False)
            ckpt['temperature'] = T_value
            ckpt['calibration_bias'] = calibration_bias
            torch.save(ckpt, save_best_path)
    
    # Free memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    splits = {"train_idx": train_idx, "val_idx": val_idx}
    return model, T_value, history, splits


# ============================================================================
# ROBUST EVALUATION
# ============================================================================

@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    T: float = 1.0,
    calibration_bias: float = 0.0,
    device: Union[str, torch.device] = None,
    batch_size: int = 256,
    return_logits: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prediction of calibrated probabilities.
    
    Args:
        model: InceptionTime model
        X: Data (N, T, F)
        T: Calibration temperature
        calibration_bias: Additive bias learned on the validation set
        device: PyTorch device
        batch_size: Batch size for inference
        return_logits: Also return the raw logits
    
    Returns:
        probas (and optionally logits)
    """
    # Device detection
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    # Make sure the model is on the right device
    model = model.to(device)
    model.eval()
    
    # Temperature validation
    if not np.isfinite(T) or T <= 0:
        warnings.warn(f"Température invalide ({T}) → fallback T=1.0")
        T = 1.0
    
    # Data validation
    if X.ndim != 3:
        raise ValueError(f"X doit être (N,T,F), reçu shape {X.shape}")
    
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf")
    
    # Dataset temporaire
    y_dummy = np.zeros(len(X))
    ds = TimeSeriesDataset(X, y_dummy)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )
    
    all_probs = []
    all_logits = [] if return_logits else None
    
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        
        # Logits check
        if not torch.isfinite(logits).all():
            warnings.warn("Non-finite logits detected in a batch")
        
        calibrated_logits = logits / T + calibration_bias
        probs = torch.sigmoid(calibrated_logits)
        
        all_probs.append(probs.cpu().numpy())
        if return_logits:
            all_logits.append(logits.cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    
    if return_logits:
        logits = np.concatenate(all_logits, axis=0)
        return probs, logits
    
    return probs


def evaluate_on_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    device: Union[str, torch.device] = None,
    batch_size: int = 256,
    return_details: bool = False
) -> Union[Tuple[float, float, float], Tuple[float, float, float, dict]]:
    """
    Robust evaluation on the test set with full checks.
    
    Args:
        X_test: Test data (N, T, F)
        y_test: Test labels
        checkpoint_path: Checkpoint path
        device: PyTorch device (None = auto-detection)
        batch_size: Batch size for inference
        return_details: Return details (probas, logits, etc.)
    
    Returns:
        (auc, brier, T) or (auc, brier, T, details)
    
    Raises:
        ValueError: If dimension incompatibility or invalid labels
        FileNotFoundError: If the checkpoint cannot be found
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for evaluation (pip install scikit-learn)")
    
    # Model loading
    model, init_args, T, calibration_bias = load_model_from_checkpoint(checkpoint_path, device)
    
    # CORRECTION: ensure device is properly defined
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)
    
    # Temperature fallback if missing
    if T is None or not np.isfinite(T) or T <= 0:
        warnings.warn(f"Température invalide ou absente ({T}) → T=1.0 (non calibré)")
        T = 1.0
    
    print(f"✓ Modèle chargé : {checkpoint_path}")
    print(f"  Device        : {device}")
    print(f"  Temperature T : {T:.4f}")
    print(f"  Calibration bias : {calibration_bias:.4f}")
    
    # Validation dimensions
    if X_test.ndim != 3:
        raise ValueError(f"X_test doit être (N,T,F), reçu shape {X_test.shape}")
    
    F_required = int(init_args.get('in_channels', -1))
    if F_required > 0 and X_test.shape[2] != F_required:
        raise ValueError(
            f"Mismatch features : modèle attend F={F_required}, "
            f"X_test a F={X_test.shape[2]}"
        )
    
    # Validation labels
    if y_test.ndim == 2:
        y_test = y_test.reshape(-1)
    
    y_flat = y_test.astype(float)
    uniq = np.unique(y_flat)
    
    if not set(uniq).issubset({0.0, 1.0}):
        raise ValueError(f"y_test doit être binaire {{0,1}}, reçu valeurs uniques : {uniq}")
    
    if len(uniq) < 2:
        raise ValueError("y_test contains only a single class -> AUC undefined")
    
    if not (np.isfinite(X_test).all() and np.isfinite(y_flat).all()):
        raise ValueError("X_test or y_test contains NaN/Inf")
    
    # Prediction (CORRECTION: pass device explicitly)
    print(f"Prédiction sur {len(X_test)} exemples...")
    try:
        p_test, logits = predict_proba(
            model, X_test, T=T, calibration_bias=calibration_bias, device=device,
            batch_size=batch_size, return_logits=True
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction : {e}") from e
    
    # Prediction check before masking
    print(f"  Stats prédictions :")
    print(f"    - min  : {np.min(p_test):.6f}")
    print(f"    - max  : {np.max(p_test):.6f}")
    print(f"    - mean : {np.mean(p_test):.6f}")
    print(f"    - NaN  : {np.isnan(p_test).sum()}")
    print(f"    - Inf  : {np.isinf(p_test).sum()}")
    
    if not np.isfinite(p_test).any():
        raise RuntimeError(
            "ALL predictions are NaN/Inf! "
            "Check that the model and the data are on the same device. "
            f"Device utilisé : {device}"
        )
    
    # Safety mask
    mask = np.isfinite(p_test) & np.isfinite(y_flat)
    n_invalid = (~mask).sum()
    n_valid = mask.sum()
    
    print(f"  Exemples valides : {n_valid} / {len(y_flat)}")
    
    if n_invalid > 0:
        warnings.warn(f"{n_invalid} exemples avec prédictions non finies (exclus des métriques)")
    
    if n_valid == 0:
        # Detailed debug
        print("\n" + "="*60)
        print("CRITICAL ERROR: No valid prediction")
        print("="*60)
        print(f"Device du modèle    : {device}")
        print(f"Shape X_test        : {X_test.shape}")
        print(f"Dtype X_test        : {X_test.dtype}")
        print(f"X_test contient NaN : {np.isnan(X_test).any()}")
        print(f"X_test contient Inf : {np.isinf(X_test).any()}")
        print(f"y_test unique       : {np.unique(y_flat)}")
        print(f"Température T       : {T}")
        print(f"p_test shape        : {p_test.shape}")
        print(f"p_test all NaN      : {np.isnan(p_test).all()}")
        print(f"p_test all Inf      : {np.isinf(p_test).all()}")
        print(f"logits shape        : {logits.shape}")
        print(f"logits stats        : min={np.min(logits):.2f}, max={np.max(logits):.2f}, mean={np.mean(logits):.2f}")
        print("="*60)
        
        raise ValueError(
            "No valid example after filtering! "
            "All predictions are NaN/Inf. "
            "See the statistics above to diagnose."
        )
    
    # Metrics
    auc = roc_auc_score(y_flat[mask], p_test[mask])
    brier = brier_score_loss(y_flat[mask], p_test[mask])
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS D'ÉVALUATION")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {os.path.basename(checkpoint_path)}")
    print(f"  Device        : {device}")
    print(f"  Température T : {T:.6f}")
    print(f"  AUC-ROC       : {auc:.6f}")
    print(f"  Brier Score   : {brier:.6f}")
    print(f"  N test        : {mask.sum()} / {len(y_flat)}")
    if n_invalid > 0:
        print(f"  N invalides   : {n_invalid}")
    print(f"{'='*60}\n")
    
    if return_details:
        details = {
            'p_test': p_test,
            'logits': logits,
            'y_test': y_flat,
            'mask': mask,
            'init_args': init_args,
            'checkpoint_path': checkpoint_path,
            'device': str(device)
        }
        return auc, brier, T, details
    
    return auc, brier, T


def recompute_temperature(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: Union[str, torch.device] = None,
    batch_size: int = 256
) -> float:
    """
    Recalculates the temperature on a new validation set.
    Useful to recalibrate without retraining.
    
    Args:
        model: InceptionTime model
        X_val: Validation data (N, T, F)
        y_val: Validation labels
        device: PyTorch device
        batch_size: Batch size
    
    Returns:
        Calibrated temperature
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model.eval()
    
    # Dataset
    ds = TimeSeriesDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    # Logits retrieval
    logits_val = _gather_logits(model, loader, device)
    y = torch.from_numpy(y_val.reshape(-1)).float().to(device)
    
    if logits_val.ndim == 1:
        logits_val = logits_val.unsqueeze(-1)
    
    # Calibration
    calibrator = TemperatureCalibrator(init_T=1.0).to(device)
    _ = calibrator.fit(logits_val, y, max_iter=200)
    T = float(calibrator.T.detach().cpu())
    
    if not np.isfinite(T) or T <= 0:
        warnings.warn(f"Recalibration a produit T={T} invalide → fallback T=1.0")
        T = 1.0
    
    print(f"✓ Nouvelle température calibrée : T = {T:.4f}")
    return T


# ============================================================================
# SIMPLIFIED API
# ============================================================================

__all__ = [
    # Classes
    'InceptionModel',
    'TemperatureCalibrator',
    'TimeSeriesDataset',
    
    # Fonctions principales
    'train_inception_time',
    'fine_tune_inception_time',
    'evaluate_on_test',
    'predict_proba',
    
    # Utils
    'load_model_from_checkpoint',
    'recompute_temperature',
    'stratified_train_val_indices',
    'compute_pos_weight_from_indices',
]


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the module.
    """
    print("InceptionTime - Module complet")
    print("=" * 60)
    print("\nExemple d'utilisation :\n")
    
    example_code = '''
# 1. Training from scratch
from inception_time import train_inception_time

model, T, history, splits = train_inception_time(
    X_train, y_train,
    epochs=100,
    patience=10,
    save_best_path="models/inception_mimic.pt"
)

# 2. Fine-tuning on a new dataset
from inception_time import fine_tune_inception_time

model_ft, T_ft, hist_ft, splits_ft = fine_tune_inception_time(
    "models/inception_mimic.pt",
    X_ecmo, y_ecmo,
    last_k_blocks=2,      # Unfreezes the last 2 blocks
    epochs=50,
    save_best_path="models/inception_ecmo_ft.pt"
)

# 3. Evaluation
from inception_time import evaluate_on_test

auc, brier, T = evaluate_on_test(
    X_test, y_test,
    "models/inception_ecmo_ft.pt"
)

# 4. Simple prediction
from inception_time import predict_proba, load_model_from_checkpoint

model, _, T = load_model_from_checkpoint("models/inception_ecmo_ft.pt")
probas = predict_proba(model, X_new, T=T)
    '''
    
    print(example_code)
    print("\n" + "=" * 60)
    print("Module ready to be imported!")
    print("Save this file as: inception_time.py")