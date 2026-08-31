"""
Vanilla Transformer (TST) for binary time-series classification
==============================================================

Complete standalone module providing:
- Vanilla Transformer Encoder architecture (Time Series Transformer)
- Adaptive sinusoidal positional encoding for short sequences (24 time steps)
- Global Average Pooling for temporal aggregation
- Training with early stopping, class weighting, and temperature calibration
- Shallow fine-tuning (freezing the first attention layers)
- Robust evaluation with calibrated metrics (AUC, Brier Score)

Usage:
    from vanilla_transformer import train_vanilla_transformer, fine_tune_vanilla_transformer, evaluate_on_test

    # Training from scratch
    model, T, history, splits = train_vanilla_transformer(X_train, y_train)

    # Fine-tuning
    model_ft, T_ft, hist_ft, splits_ft = fine_tune_vanilla_transformer(
        "best_transformer.pt", X_new, y_new, last_k_layers=1
    )

    # Evaluation
    auc, brier, T = evaluate_on_test(X_test, y_test, "best_transformer.pt")

Author: Consistent adaptation with the InceptionTime API
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
    warnings.warn("tqdm non disponible - pas de barre de progression")

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn non disponible - métriques d'évaluation indisponibles")


# ============================================================================
# VANILLA TRANSFORMER (TST) ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used to inject temporal order."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VanillaTransformerModel(nn.Module):
    """
    Standard Transformer Encoder model for binary time-series classification.

    Architecture:
    Input (N, T, F) -> Linear Projection (N, T, d_model) -> Positional Encoding
    -> N x TransformerEncoderLayer -> Global Average Pooling -> Linear Head -> Logit (N, 1)
    """
    
    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 100,
        num_pred_classes: int = 1
    ) -> None:
        super().__init__()
        
        # Save arguments for model reconstruction
        self.input_args = {
            'in_channels': in_channels,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_len': max_len,
            'num_pred_classes': num_pred_classes,
        }
        
        # 1. Linear projection of input features to d_model
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        # 3. Transformer attention layers (encoder only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 4. Linear classification head
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=num_pred_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, T, F)
        x = self.input_projection(x)      # (N, T, d_model)
        x = self.pos_encoder(x)          # (N, T, d_model)
        x = self.transformer_encoder(x)  # (N, T, d_model)
        
        # Global Average Pooling over the time axis (T)
        x = x.mean(dim=1)                # (N, d_model)
        
        return self.linear(x)            # (N, num_pred_classes)


# ============================================================================
# TEMPERATURE CALIBRATION
# ============================================================================

class TemperatureCalibrator(nn.Module):
    """
    Affine calibrator for binary classification.
    calibrated_logits = logits / T + bias
    """

    def __init__(self, init_T: float = 1.0, init_bias: float = 0.0) -> None:
        super().__init__()
        if init_T <= 0:
            raise ValueError("init_T doit être strictement positif")

        self.log_T = nn.Parameter(
            torch.tensor(math.log(init_T), dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor(init_bias, dtype=torch.float32)
        )

    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T + self.bias

    def fit(
        self,
        logits_val: torch.Tensor,
        y_val: torch.Tensor,
        max_iter: int = 200
    ) -> float:
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
# DATASET AND UTILITIES
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series with shape (N, T, F)."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, "X doit être (N, T, F)"
        if y.ndim == 2:
            y = y.reshape(-1)
        
        # With batch_first=True, Transformers keep the (N, T, F) layout
        self.X = torch.from_numpy(X).float()
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
    """Compute pos_weight for BCEWithLogitsLoss (class balancing)."""
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
    """Collect logits from a DataLoader."""
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
# TRAINING FROM SCRATCH
# ============================================================================

def train_vanilla_transformer(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float = 0.2,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad: Optional[float] = 1.0,
    use_scheduler: bool = True,
    calibrate: bool = True,
    save_best_path: Optional[str] = "best_vanilla_transformer.pt",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    progress: bool = True,
    seed: int = 42,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[VanillaTransformerModel, float, Dict[str, list], Dict[str, np.ndarray]]:
    """Train a Vanilla Transformer from scratch."""
    
    assert X.shape[0] == y.reshape(-1).shape[0], "Mismatch N entre X et y"
    assert X.shape[2] > 0, "X doit avoir au moins 1 feature"
    assert X.ndim == 3, f"X doit être (N,T,F), reçu shape {X.shape}"
    
    device = torch.device(device)
    
    # Determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gen = torch.Generator()
    gen.manual_seed(seed)

    if X_val is not None and y_val is not None:
        assert X_val.ndim == 3, f"X_val doit être (N,T,F), reçu shape {X_val.shape}"
        assert X_val.shape[0] == y_val.reshape(-1).shape[0], "Mismatch N entre X_val et y_val"
        assert X_val.shape[2] == X.shape[2], "Mismatch de features entre X et X_val"
        
        train_ds = TimeSeriesDataset(X, y)
        val_ds = TimeSeriesDataset(X_val, y_val)
        train_idx = np.arange(len(X))
        val_idx = np.arange(len(X_val))
    else:
        train_idx, val_idx = stratified_train_val_indices(y, val_ratio=val_ratio, seed=seed)
        ds = TimeSeriesDataset(X, y)
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
    
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
    
    # Transformer model
    in_channels = X.shape[2]
    model = VanillaTransformerModel(
        in_channels=in_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=X.shape[1] + 10,  # Safety check for T=24
        num_pred_classes=1
    ).to(device)
    
    # Loss with class weighting
    pos_w = compute_pos_weight_from_indices(y, train_idx)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device) if pos_w is not None else None
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}
    patience_count = 0
    
    if progress and TQDM_AVAILABLE:
        pbar = tqdm(range(1, epochs + 1), desc="Training Transformer", leave=True)
    else:
        pbar = range(1, epochs + 1)
    
    for epoch in pbar:
        # TRAINING
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
        
        # VALIDATION
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
        
        if scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if progress and TQDM_AVAILABLE:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                lr=f"{current_lr:.2e}"
            )
        
        # EARLY STOPPING
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            
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
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Best model loaded (epoch {best_epoch}, val_loss={best_val:.6f})")
    
    # TEMPERATURE CALIBRATION
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
        
        print(f"✓ Calibration: T = {T_value:.4f}, bias = {calibration_bias:.4f}")
        
        if save_best_path is not None and os.path.exists(save_best_path):
            ckpt = torch.load(save_best_path, map_location='cpu', weights_only=False)
            ckpt['temperature'] = T_value
            ckpt['calibration_bias'] = calibration_bias
            torch.save(ckpt, save_best_path)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    splits = {"train_idx": train_idx, "val_idx": val_idx}
    return model, T_value, history, splits


# ============================================================================
# SHALLOW FINE-TUNING
# ============================================================================

def set_trainable_last_k_layers(
    model: VanillaTransformerModel,
    last_k_layers: int = 1,
    train_linear: bool = True
):
    """Freeze the first Transformer layers and unfreeze the final block."""
    num_layers = len(model.transformer_encoder.layers)
    k = max(0, min(last_k_layers, num_layers))
    
    for p in model.parameters():
        p.requires_grad = False
    
    def force_eval_mode(module, input):
        module.eval()
    
    for i in range(0, num_layers - k):
        layer = model.transformer_encoder.layers[i]
        layer.eval()
        layer.register_forward_pre_hook(force_eval_mode)
    
    for i in range(num_layers - k, num_layers):
        layer = model.transformer_encoder.layers[i]
        layer.train()
        for p in layer.parameters():
            p.requires_grad = True
            
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
) -> Tuple[VanillaTransformerModel, dict, Optional[float], float]:
    """Load the Transformer model from a checkpoint."""
    device = torch.device(device) if device is not None else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trouvé : {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    init_args = ckpt['init_args']
    state_dict = ckpt.get('state_dict') or ckpt.get('model_state_dict')
    T = ckpt.get('temperature', None)
    calibration_bias = float(ckpt.get('calibration_bias', 0.0))
    
    model = VanillaTransformerModel(**init_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, init_args, T, calibration_bias


def fine_tune_vanilla_transformer(
    model_or_ckpt: Union[str, VanillaTransformerModel],
    X_ft: np.ndarray,
    y_ft: np.ndarray,
    *,
    last_k_layers: int = 1,
    train_linear: bool = True,
    reinit_linear: bool = False,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
    min_delta: float = 0.0,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    clip_grad: Optional[float] = 0.5,
    use_scheduler: bool = True,
    calibrate: bool = True,
    save_best_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    progress: bool = True,
    seed: int = 42,
) -> Tuple[VanillaTransformerModel, float, Dict[str, list], Dict[str, np.ndarray]]:
    """Perform shallow fine-tuning of the Vanilla Transformer."""
    
    device = torch.device(device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    if isinstance(model_or_ckpt, str):
        model, init_args, _T, _calibration_bias = load_model_from_checkpoint(model_or_ckpt, device)
        print(f"✓ Model loaded from {model_or_ckpt}")
    else:
        model = model_or_ckpt.to(device)
        init_args = getattr(model, 'input_args', None) or {}
    
    assert X_ft.ndim == 3, f"X_ft doit être (N,T,F), reçu {X_ft.shape}"
    F_in = X_ft.shape[2]
    if init_args and 'in_channels' in init_args:
        F_expected = init_args['in_channels']
        assert F_in == F_expected, f"Mismatch features : modèle attend {F_expected}, X_ft a {F_in}"
    
    if reinit_linear:
        print("Reinitializing the linear layer...")
        model.linear.reset_parameters()
    
    set_trainable_last_k_layers(
        model,
        last_k_layers=last_k_layers,
        train_linear=train_linear
    )
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"✓ Fine-tuning: {n_trainable:,} / {n_total:,} trainable parameters ({100*n_trainable/n_total:.1f}%)")
    
    ds = TimeSeriesDataset(X_ft, y_ft)
    train_idx, val_idx = stratified_train_val_indices(y_ft, val_ratio=val_ratio, seed=seed)
    
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        generator=gen
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )
    
    pos_w = compute_pos_weight_from_indices(y_ft, train_idx)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_w.to(device) if pos_w is not None else None
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}
    patience_count = 0
    
    if progress and TQDM_AVAILABLE:
        pbar = tqdm(range(1, epochs + 1), desc="Fine-tuning Transformer", leave=True)
    else:
        pbar = range(1, epochs + 1)
    
    for epoch in pbar:
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
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=clip_grad
                )
            
            optimizer.step()
            
            bs = xb.size(0)
            run_loss += float(loss.detach().cpu()) * bs
            n_obs += bs
        
        train_loss = run_loss / max(1, n_obs)
        
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
        
        if scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if progress and TQDM_AVAILABLE:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                lr=f"{current_lr:.2e}"
            )
        
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
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Best model loaded (epoch {best_epoch}, val_loss={best_val:.6f})")
    
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
        
        print(f"✓ Calibration: T = {T_value:.4f}, bias = {calibration_bias:.4f}")
        
        if save_best_path is not None and os.path.exists(save_best_path):
            ckpt = torch.load(save_best_path, map_location='cpu', weights_only=False)
            ckpt['temperature'] = T_value
            ckpt['calibration_bias'] = calibration_bias
            torch.save(ckpt, save_best_path)
    
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
    """Predict calibrated probabilities with the Transformer."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    if not np.isfinite(T) or T <= 0:
        warnings.warn(f"Température invalide ({T}) → fallback T=1.0")
        T = 1.0
    
    if X.ndim != 3:
        raise ValueError(f"X doit être (N,T,F), reçu shape {X.shape}")
    
    if not np.isfinite(X).all():
        raise ValueError("X contient des NaN/Inf")
    
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
        
        if not torch.isfinite(logits).all():
            warnings.warn("Logits non finis détectés dans un batch")
        
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
    """Evaluate the model on the test set."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn requis pour l'évaluation (pip install scikit-learn)")
    
    model, init_args, T, calibration_bias = load_model_from_checkpoint(checkpoint_path, device)
    
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)
    
    if T is None or not np.isfinite(T) or T <= 0:
        warnings.warn(f"Température invalide ou absente ({T}) → T=1.0 (non calibré)")
        T = 1.0
    
    print(f"✓ Model loaded: {checkpoint_path}")
    print(f"  Device           : {device}")
    print(f"  Temperature T    : {T:.4f}")
    print(f"  Calibration bias : {calibration_bias:.4f}")
    
    if X_test.ndim != 3:
        raise ValueError(f"X_test doit être (N,T,F), reçu shape {X_test.shape}")
    
    F_required = int(init_args.get('in_channels', -1))
    if F_required > 0 and X_test.shape[2] != F_required:
        raise ValueError(
            f"Mismatch features : modèle attend F={F_required}, "
            f"X_test a F={X_test.shape[2]}"
        )
    
    if y_test.ndim == 2:
        y_test = y_test.reshape(-1)
    
    y_flat = y_test.astype(float)
    uniq = np.unique(y_flat)
    
    if not set(uniq).issubset({0.0, 1.0}):
        raise ValueError(f"y_test doit être binaire {{0,1}}, reçu valeurs uniques : {uniq}")
    
    if len(uniq) < 2:
        raise ValueError("y_test ne contient qu'une seule classe → AUC non définie")
    
    if not (np.isfinite(X_test).all() and np.isfinite(y_flat).all()):
        raise ValueError("X_test ou y_test contient des NaN/Inf")
    
    print(f"Predicting on {len(X_test)} samples...")
    try:
        p_test, logits = predict_proba(
            model, X_test, T=T, calibration_bias=calibration_bias, device=device,
            batch_size=batch_size, return_logits=True
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction : {e}") from e
    
    print(f"  Prediction statistics:")
    print(f"    - min  : {np.min(p_test):.6f}")
    print(f"    - max  : {np.max(p_test):.6f}")
    print(f"    - mean : {np.mean(p_test):.6f}")
    
    mask = np.isfinite(p_test) & np.isfinite(y_flat)
    n_valid = mask.sum()
    
    if n_valid == 0:
        raise ValueError("Aucun exemple valide après filtrage ! Toutes les prédictions sont NaN/Inf.")
    
    auc = roc_auc_score(y_flat[mask], p_test[mask])
    brier = brier_score_loss(y_flat[mask], p_test[mask])
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (VANILLA TRANSFORMER)")
    print(f"{'='*60}")
    print(f"  Checkpoint    : {os.path.basename(checkpoint_path)}")
    print(f"  Device        : {device}")
    print(f"  Temperature T : {T:.6f}")
    print(f"  AUC-ROC       : {auc:.6f}")
    print(f"  Brier Score   : {brier:.6f}")
    print(f"  N test        : {mask.sum()} / {len(y_flat)}")
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


# ============================================================================
# SIMPLIFIED API
# ============================================================================

__all__ = [
    'VanillaTransformerModel',
    'TemperatureCalibrator',
    'TimeSeriesDataset',
    'train_vanilla_transformer',
    'fine_tune_vanilla_transformer',
    'evaluate_on_test',
    'predict_proba',
    'load_model_from_checkpoint',
    'stratified_train_val_indices',
    'compute_pos_weight_from_indices',
]


if __name__ == "__main__":
    print("Vanilla Transformer (TST) - Complete module")
    print("=" * 60)
    print("Save this file as: vanilla_transformer.py")