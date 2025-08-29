import torch
from threestudio.utils.ops import get_cam_info_gaussian
from torch.cuda.amp import autocast

from ..geometry.gaussian_base import BasicPointCloud, Camera


class GaussianBatchRenderer:
    def batch_forward(self, batch):
        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        pred_normals = []
        depths = []
        masks = []

        ref_masks = []
        ref_depths = []
        ref_normals = []
        ref_renders = []
        ref_viewspace_points = []
        ref_visibility_filter = []
        ref_radii = []
        ref_pred_normals = []
        
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.forward(
                    viewpoint_cam, self.background_tensor, **batch
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if (
                    render_pkg.__contains__("pred_normal")
                    and render_pkg["pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["pred_normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])

                if render_pkg.__contains__("ref_render"):
                    ref_renders.append(render_pkg["ref_render"])
                if render_pkg.__contains__("ref_viewspace_points"):
                    ref_viewspace_points.append(render_pkg["ref_viewspace_points"])
                if render_pkg.__contains__("ref_visibility_filter"):
                    ref_visibility_filter.append(render_pkg["ref_visibility_filter"])
                if render_pkg.__contains__("ref_radii"):
                    ref_radii.append(render_pkg["ref_radii"])

                if render_pkg.__contains__("ref_normal"):
                    ref_normals.append(render_pkg["ref_normal"])        
                if (
                    render_pkg.__contains__("ref_pred_normal")
                    and render_pkg["ref_pred_normal"] is not None
                ):
                    pred_normals.append(render_pkg["ref_pred_normal"])
                if render_pkg.__contains__("ref_depth"):
                    ref_depths.append(render_pkg["ref_depth"])
                if render_pkg.__contains__("ref_mask"):
                    ref_masks.append(render_pkg["ref_mask"])

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(pred_normals) > 0:
            outputs.update(
                {
                    "comp_pred_normal": torch.stack(pred_normals, dim=0).permute(
                        0, 2, 3, 1
                    ),
                }
            )
        if len(depths) > 0:
            outputs.update(
                {
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(masks) > 0:
            outputs.update(
                {
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )

        # Add the reference renders to outputs
        if len(ref_renders) > 0:
            outputs.update(
                {
                    "ref_rgb": torch.stack(ref_renders, dim=0).permute(0, 2, 3, 1),
                    "ref_viewspace_points": ref_viewspace_points,
                    "ref_visibility_filter": ref_visibility_filter,
                    "ref_radii": ref_radii,
                }
            )
        if len(ref_normals) > 0:
            outputs.update(
                {
                    "ref_normal": torch.stack(ref_normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(ref_pred_normals) > 0:
            outputs.update(
                {
                    "ref_pred_normal": torch.stack(ref_pred_normals, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(ref_depths) > 0:
            outputs.update(
                {
                    "ref_depth": torch.stack(ref_depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if len(ref_masks) > 0:
            outputs.update(
                {
                    "ref_mask": torch.stack(ref_masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        return outputs
