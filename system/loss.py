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