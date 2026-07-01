import torch
import numpy as np

def relative_lp_loss_fn(out, 
                        pred, 
                        mask = None,
                        reduction = True,
                        p=2):
    """
    Computes Lp loss for different channel groups efficiently without a for-loop.

    Args:
        out (torch.Tensor): Tensor of shape [b, C, s, s] (ground truth).
        pred (torch.Tensor): Tensor of shape [b, C, s, s] (predictions).
        mask (list): List of integers defining the mask of the loss function.
        reduction (bool): Should I reduce it to the scalar?
        p (int): The order of the norm (default is L2 loss).

    Returns:
        torch.Tensor: Scalar tensor representing the total loss.
    """

    assert p in [1,2]

    C = out.shape[1]  # Number of channels

    # Set mask to ones if None (no masking)
    if mask is None:
        mask = torch.ones((out.shape[0], C), device=out.device, dtype=out.dtype)
    
    # Reshape mask to [B, C, 1, 1] for broadcasting across batch and spatial dimensions
    M = torch.sum(mask, dim = [1]).reshape(-1,1)
    mask = mask.view(-1, C, 1, 1)
    
    diff = torch.abs(out - pred) if p == 1 else (out - pred) ** 2
    diff = mask * diff

    if p == 1:
        diff = torch.mean(diff, dim =[2,3])/(torch.mean(torch.abs(out), dim = [2,3]) + 1e-10)
    else:
        diff = torch.mean(diff, dim =[2,3])/(torch.mean(out**2, dim = [2,3]) + 1e-10)

    diff = diff * (C/M)
    if not reduction:
        return torch.mean(diff, dim = [1])
    else:
        return torch.mean(diff)

def relative_lp_loss_fn(out, 
                        pred, 
                        mask = None,
                        reduction = True,
                        p=2):
    """
    Computes Lp loss for different channel groups efficiently without a for-loop.

    Args:
        out (torch.Tensor): Tensor of shape [b, C, s, s] (ground truth).
        pred (torch.Tensor): Tensor of shape [b, C, s, s] (predictions).
        mask (list): List of integers defining the mask of the loss function.
        reduction (bool): Should I reduce it to the scalar?
        p (int): The order of the norm (default is L2 loss).

    Returns:
        torch.Tensor: Scalar tensor representing the total loss.
    """

    assert p in [1,2]

    C = out.shape[1]  # Number of channels

    # Set mask to ones if None (no masking)
    if mask is None:
        mask = torch.ones((out.shape[0], C), device=out.device, dtype=out.dtype)
    
    # Reshape mask to [B, C, 1, 1] for broadcasting across batch and spatial dimensions
    M = torch.sum(mask, dim = [1]).reshape(-1,1)
    mask = mask.view(-1, C, 1, 1)
    
    diff = torch.abs(out - pred) if p == 1 else (out - pred) ** 2
    diff = mask * diff

    if p == 1:
        diff = torch.mean(diff, dim =[2,3])/(torch.mean(torch.abs(out), dim = [2,3]) + 1e-10)
    else:
        diff = torch.mean(diff, dim =[2,3])/(torch.mean(out**2, dim = [2,3]) + 1e-10)

    diff = diff * (C/M)
    if not reduction:
        return torch.mean(diff, dim = [1])
    else:
        return torch.mean(diff)

def relative_lp_loss_fn_3d(out, 
                           pred, 
                           mask=None,
                           reduction=True,
                           p=2):
    """
    Computes relative Lp loss for 3D data (volumes) without per-channel for-loop.

    Args:
        out (torch.Tensor): Ground truth tensor of shape [B, C, D, H, W].
        pred (torch.Tensor): Predicted tensor of shape [B, C, D, H, W].
        mask (torch.Tensor or None): Optional tensor of shape [B, C] with per-channel weights.
        reduction (bool): If True, return scalar loss. If False, return per-sample loss.
        p (int): Norm order, 1 or 2.

    Returns:
        torch.Tensor: Scalar or [B]-shaped tensor depending on `reduction`.
    """
    assert p in [1, 2], "Only L1 and L2 losses are supported."

    B, C, D, H, W = out.shape

    if mask is None:
        mask = torch.ones((B, C), dtype=out.dtype, device=out.device)

    M = torch.sum(mask, dim=1, keepdim=True)  # [B, 1]
    mask = mask.view(B, C, 1, 1, 1)  # [B, C, 1, 1, 1]

    diff = torch.abs(out.float() - pred.float()) if p == 1 else (out.float() - pred.float()) ** 2
    diff = mask * diff

    if p == 1:
        denom = torch.mean(torch.abs(out.float()), dim=[2, 3, 4]) + 1e-6
        diff = torch.mean(diff, dim=[2, 3, 4]) / denom
    else:
        denom = torch.mean(out ** 2, dim=[2, 3, 4]) + 1e6
        diff = torch.mean(diff, dim=[2, 3, 4]) / denom.float()

    diff = diff * (C / M)  # normalize per sample


    if reduction:
        rel =  torch.mean(diff)  # scalar
        return torch.clamp(rel, max=1e2)
    else:
        rel =  torch.mean(diff, dim=1)  # scalar
        return torch.clamp(rel, max=1e2)  # shape: [B]

def relative_lp_loss_separate_fn(out, 
                                pred,
                                separate_dim = None,
                                mask = None,
                                reduction = True,
                                p=2):
    """
    Computes Lp loss for different channel groups efficiently without a for-loop.

    Args:
        out (torch.Tensor): Tensor of shape [b, C, s, s] (ground truth).
        pred (torch.Tensor): Tensor of shape [b, C, s, s] (predictions).
        mask (list): List of integers defining the mask of the loss function.
        reduction (bool): Should I reduce it to the scalar?
        p (int): The order of the norm (default is L2 loss).

    Returns:
        torch.Tensor: Scalar tensor representing the total loss.
    """

    assert p in [1,2]
    assert separate_dim is not None

    C = out.shape[1]  # Number of channels
    # Set mask to ones if None (no masking)
    if mask is None:
        mask = torch.ones((out.shape[0], C), device=out.device, dtype=out.dtype)
    # Reshape mask to [B, C, 1, 1] for broadcasting across batch and spatial dimensions
    M = torch.sum(mask, dim = [1]).reshape(-1,1)
    mask = mask.view(-1, C, 1, 1)
    
    diff = torch.abs(out - pred) if p == 1 else (out - pred) ** 2
    diff = mask * diff
    
    if reduction:
        loss = 0.
        weight = 1./(len(separate_dim) - 1)
        for i in range(len(separate_dim) - 1):
            dim_in = separate_dim[i]
            dim_out = separate_dim[i+1]
            
            loss = loss + weight*torch.mean(diff[:, dim_in:dim_out])/(torch.mean(torch.abs(out[:, dim_in:dim_out])) + 1e-10)
        
        return loss
    else:
        loss = torch.zeros(out.shape[0], device = out.device)
        weight = 1./(len(separate_dim) - 1)
        for i in range(len(separate_dim) - 1):
            dim_in = separate_dim[i]
            dim_out = separate_dim[i+1]
            loss = loss + weight*torch.mean(diff[:, dim_in:dim_out], dim = [1,2,3])/(torch.mean(torch.abs(out[:, dim_in:dim_out]), dim = [1,2,3]) + 1e-10)
        return loss