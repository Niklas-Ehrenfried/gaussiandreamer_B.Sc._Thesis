import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from threestudio.utils.ops import get_cam_info_gaussian
from pytorch3d.loss import chamfer_distance

def gpu_approx_distance_transform(binary_mask: torch.Tensor, max_radius: int = 16):
    device = binary_mask.device
    N, C, H, W = binary_mask.shape
    assert C == 1

    fg = (binary_mask > 0.5).to(dtype=torch.uint8)  # (N,1,H,W)
    dist = torch.full((N, 1, H, W), -1, dtype=torch.int16, device=device)

    dist[fg.bool()] = 0
    coverage = fg.clone().to(dtype=torch.uint8)

    for step in range(1, max_radius + 1):
        coverage = F.max_pool2d(coverage.float(), kernel_size=3, stride=1, padding=1).to(dtype=torch.uint8)
        newly = (coverage == 1) & (dist == -1)
        if not newly.any():
            break
        dist[newly] = step

    remaining_mask = (dist == -1)
    if remaining_mask.any():
        dist[remaining_mask] = max_radius + 1

    dist_f = dist.to(dtype=torch.float32)
    return dist_f

def gpu_distance_smoothing(mask: torch.Tensor, sigma: float = 2.0, max_radius: int = 16):
    binary = (mask > 0.5).float()

    dist = gpu_approx_distance_transform(binary, max_radius=max_radius)
    edt_norm = dist / (sigma + 1e-6)
    smooth = torch.exp(-0.5 * edt_norm.pow(2))
    return smooth


# --------------------
# Projection helper
# -------------------
def project_points_with_get_cam_info(
    pts_world: torch.Tensor,
    c2w: torch.Tensor,
    fovy,
    image_size: Tuple[int, int],
    znear: float = 0.1,
    zfar: float = 100.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or pts_world.device
    B, P, _ = pts_world.shape
    H, W = image_size

    grids = []
    depths = []
    for b in range(B):
        fovy_b = fovy[b] if (torch.is_tensor(fovy) and fovy.numel() > 1) else fovy
        w2c, proj, cam_p = get_cam_info_gaussian(c2w=c2w[b], fovx=fovy_b, fovy=fovy_b, znear=znear, zfar=zfar)
        w2c = torch.as_tensor(w2c, dtype=torch.float32, device=device)
        proj = torch.as_tensor(proj, dtype=torch.float32, device=device)

        pts = pts_world[b]
        ones = torch.ones((P, 1), dtype=pts.dtype, device=device)
        pts_h = torch.cat([pts, ones], dim=-1).unsqueeze(-1)

        clip = (proj @ w2c @ pts_h).squeeze(-1)
        w = clip[:, 3:4].clamp(min=1e-9)
        ndc = clip[:, :3] / w
        xy = ndc[:, :2]

        cam_pts_h = (w2c @ pts_h).squeeze(-1)
        cam_xyz = cam_pts_h[:, :3]
        cam_z = cam_xyz[:, 2]

        grids.append(xy)
        depths.append(cam_z)

    grid = torch.stack(grids, dim=0)
    cam_depth = torch.stack(depths, dim=0)
    return grid, cam_depth


# ----------------------------------
# Sampling from Gaussian geometry
# ----------------------------------
def sample_points_from_gaussian_geometry(
    geometry,
    pts_per_gaussian: int = 16,
    device: Optional[torch.device] = None,
    max_points: Optional[int] = None,
) -> torch.Tensor:
    device = device or next(iter(geometry.parameters())).device if any(True for _ in geometry.parameters()) else geometry.get_xyz.device
    centers = geometry.get_xyz.to(device)
    scales = geometry.get_scaling.to(device)
    N = centers.shape[0]

    if scales.ndim == 1:
        scales_vec = scales.view(-1, 1, 1)
    else:
        scales_vec = scales.view(N, 1, 3)

    eps = torch.randn(N, pts_per_gaussian, 3, device=device)
    pts = centers.unsqueeze(1) + eps * scales_vec
    pts = pts.view(1, -1, 3)
    if max_points is not None and pts.shape[1] > max_points:
        idx = torch.randperm(pts.shape[1], device=device)[:max_points]
        pts = pts[:, idx, :]
    return pts




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

# -------------------------
# Alpha silhouette  loss 
# -------------------------
def alpha_silhouette_loss(
    comp_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    weight: float = 1.0,
    reduction: str = 'mean',
    visual_log_callback=None,
    global_step: int = None
) -> torch.Tensor:
    per_pixel_sil_loss = torch.nn.functional.mse_loss(comp_mask, ref_mask, reduction='none')
    loss_sil = per_pixel_sil_loss.mean() * weight if reduction == 'mean' else per_pixel_sil_loss * weight

    if visual_log_callback is not None and global_step is not None and global_step % 100 == 0:
        comp_mask_img = comp_mask.detach().squeeze(-1)
        ref_mask_img = ref_mask.detach().squeeze(-1)
        per_pixel_sil_loss_img = per_pixel_sil_loss.detach().squeeze(-1)

        images_to_save = []
        for i in range(min(4, comp_mask_img.shape[0])):
            images_to_save.append([
                {"type": "grayscale", "img": comp_mask_img[i], "kwargs": {"cmap": None}},
                {"type": "grayscale", "img": ref_mask_img[i], "kwargs": {"cmap": None}},
                {"type": "grayscale", "img": per_pixel_sil_loss_img[i], "kwargs": {"cmap": "jet"}},
            ])
        visual_log_callback(
            f"silhouette_alpha/alpha-mask-it{global_step}.png",
            images_to_save,
            name="training_step_alpha_mask",
            step=global_step,
        )

    return loss_sil


# ------------------------
# Depth silhouette loss 
# ------------------------
def depth_silhouette_loss(
    comp_depth: torch.Tensor,
    ref_depth: torch.Tensor,
    weight: float = 1.0,
    reduction: str = 'mean',
    visual_log_callback=None,
    global_step: int = None
) -> torch.Tensor:
    per_pixel_dsil_loss = torch.nn.functional.mse_loss(comp_depth, ref_depth, reduction='none')
    loss_dsil = per_pixel_dsil_loss.mean() * weight if reduction == 'mean' else per_pixel_dsil_loss * weight

    if visual_log_callback is not None and global_step is not None and global_step % 100 == 0:
        comp_depth_img = comp_depth.detach().squeeze(-1)
        ref_depth_img = ref_depth.detach().squeeze(-1)
        per_pixel_dsil_loss_img = per_pixel_dsil_loss.detach().squeeze(-1)

        images_to_save = []
        for i in range(min(4, comp_depth_img.shape[0])):
            images_to_save.append([
                {"type": "grayscale", "img": comp_depth_img[i], "kwargs": {"cmap": "jet"}},
                {"type": "grayscale", "img": ref_depth_img[i], "kwargs": {"cmap": "jet"}},
                {"type": "grayscale", "img": per_pixel_dsil_loss_img[i], "kwargs": {"cmap": "jet"}},
            ])
        visual_log_callback(
            f"silhouette_depth/depth-mask-it{global_step}.png",
            images_to_save,
            name="training_step_depth_mask",
            step=global_step,
        )

    return loss_dsil


# -------------------------
# Masked SDS loss
# -------------------------
def masked_sds_loss(
    out: Dict[str, torch.Tensor],
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_pred: torch.Tensor,
    cfg_loss: Dict[str, Any],
    lambda_alpha: float,
    lambda_depth: float,
    lambda_masked_sds: float,
    visual_log_callback=None,
    global_step: int = None
) -> torch.Tensor:
    alpha = out['ref_mask'].squeeze(-1)
    depth = out['ref_depth'].squeeze(-1)
    max_d = depth.amax(dim=(1,2), keepdim=True)
    depth_norm = depth / (max_d + 1e-6)
    weighted_alpha = alpha * lambda_alpha
    weighted_depth = (1-depth_norm) * lambda_depth
    mask_px = (weighted_alpha + weighted_depth).clamp(0,1).unsqueeze(1)
    fill_method = cfg_loss.get('mask_fill_method', 'distance')
    if fill_method == 'distance':
        sigma = float(cfg_loss.get("distance_sigma", 8.0))
        max_r = int(cfg_loss.get("mask_distance_max_radius", 24))
        smooth_flat = gpu_distance_smoothing(mask_px, sigma=sigma, max_radius=max_r)
        mask_px_filled = torch.max(mask_px, smooth_flat)
    else:
        mask_px_filled = mask_px
    mask_px_filled = mask_px_filled.clamp(0.0, 1.0)
    mask_lat = F.interpolate(mask_px_filled, size=latents.shape[-2:], mode='bilinear', align_corners=False)
    mask_b   = mask_lat.expand_as(latents)
    resid    = (noise_pred - noise) * mask_b

    if visual_log_callback is not None and global_step is not None and global_step % 100 == 0:
        mask_px_befor_vis = mask_px.detach().squeeze(1)  # [B, H_img, W_img]
        mask_px_after_vis   = mask_px_filled.detach().squeeze(1)                   
        mask_lat_vis  = mask_lat.detach().squeeze(1)
        sds_map       = resid.detach().pow(2).sum(dim=1)
        images_to_save = []
        for i in range(min(4, mask_lat_vis.shape[0])):
            images_to_save.append([
                { "type": "grayscale", "img": mask_px_befor_vis[i],  "kwargs": {"cmap": "jet"} },
                { "type": "grayscale", "img": mask_px_after_vis[i],  "kwargs": {"cmap": "jet"} },
                { "type": "grayscale", "img": mask_lat_vis[i], "kwargs": {"cmap": "jet"} },
                { "type": "grayscale", "img": sds_map[i],     "kwargs": {"cmap": "jet"  } },
            ])
        visual_log_callback(
            f"sds_masked/masked_sds_it{global_step}.png",
            images_to_save,
            name="training_step_masked_sds",
            step=global_step,
        )
    masked_sds = resid.pow(2).sum(dim=(1,2,3)) / (mask_b.sum(dim=(1,2,3))+1e-6)
    loss_m = masked_sds.mean()
    return loss_m * lambda_masked_sds

# -------------------------
# Mask-penetration loss
# -------------------------
def mask_penetration_loss(
    out: Dict[str, torch.Tensor],
    scales: torch.Tensor,
    topk: float = 0.05,
    weight: float = 1.0,
    reduction: str = "mean",
    visual_log_callback=None,
    global_step: int = None
) -> torch.Tensor:
    comp_mask = out["comp_mask"]
    ref_mask  = out["ref_mask"]

    leakage = comp_mask * (1.0 - ref_mask)

    per_pixel_loss = leakage                 
    if reduction == "mean":
        loss_px = per_pixel_loss.mean()
    else:
        loss_px = per_pixel_loss

    # --- focus only on n largest Gaussians ---
    gauss_vol = scales.prod(dim=1)
    kth = int(max(1, gauss_vol.numel() * topk))
    top_idx = torch.topk(gauss_vol, kth, sorted=False).indices

    sel_mask = torch.zeros_like(gauss_vol, dtype=torch.bool)
    sel_mask[top_idx] = True

    if "visibility_filter" in out and len(out["visibility_filter"]) > 0:
        vis = out["visibility_filter"][0]
        sel_visible = sel_mask & vis
        if sel_visible.any():
            scale_weight = sel_visible.float().mean()
            loss_px = loss_px * scale_weight

    if visual_log_callback is not None and global_step is not None and global_step % 100 == 0:
        leakage_img   = leakage.detach().squeeze(-1)
        comp_mask_img = comp_mask.detach().squeeze(-1)
        ref_mask_img  = ref_mask.detach().squeeze(-1)

        images_to_save = []
        for i in range(min(4, leakage_img.shape[0])):
            images_to_save.append([
                {"type": "grayscale", "img": comp_mask_img[i], "kwargs": {"cmap": None}},
                {"type": "grayscale", "img": ref_mask_img[i],  "kwargs": {"cmap": None}},
                {"type": "grayscale", "img": leakage_img[i],   "kwargs": {"cmap": "jet"}},
            ])
        visual_log_callback(
            f"mask_penetration/mask-penetration-it{global_step}.png",
            images_to_save,
            name="training_step_mask_penetration",
            step=global_step,
        )

    return loss_px * weight