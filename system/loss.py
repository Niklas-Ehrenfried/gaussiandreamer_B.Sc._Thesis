import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from threestudio.utils.ops import get_cam_info_gaussian
from pytorch3d.loss import chamfer_distance


# -------------------------
# Chamfer (PyTorch3D)
# -------------------------
def chamfer_loss_p3d(pred_pc: torch.Tensor, ref_pc: torch.Tensor, weight: float = 1.0,) -> torch.Tensor:
    """
    pred_pc: (B,P,3), ref_pc: (B,Q,3)
    returns scalar (mean over batch)
    """
    loss, _ = chamfer_distance(pred_pc, ref_pc, batch_reduction="mean", point_reduction="mean")
    return loss * weight


# ---------------------------------
#  Scale outlier loss 
# ---------------------------------
def per_dim_log_smooth_l1_loss(scales: torch.Tensor, weight: float = 1.0, quantile: float = 0.98, eps: float = 1e-8) -> torch.Tensor:
    device = scales.device
    log_scales = torch.log(scales.clamp_min(eps))

    thresholds = torch.quantile(log_scales, quantile, dim=0)

    over = F.relu(log_scales - thresholds.unsqueeze(0))

    per_elem_loss = F.smooth_l1_loss(over, torch.zeros_like(over), reduction='none')
    per_gauss_loss = per_elem_loss.sum(dim=1)
    outlier_mask = (over > 0).any(dim=1)

    n_out = outlier_mask.sum()
    if n_out.item() == 0:
        return torch.tensor(0., device=device)

    return per_gauss_loss[outlier_mask].mean() * weight