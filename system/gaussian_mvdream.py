import os
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
import copy
import torch.nn.utils as nn_utils
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud
from .loss import *

@threestudio.register("gaussiandreamer-mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        lambda_chamfer: float = 0.0
        lambda_scale_outlier: float = 0.0
        lambda_alpha_silhouette_loss: float = 0.0
        lambda_depth_silhouette_loss: float = 0.0
        lambda_masked_sds: float = 0.0
        lambda_alpha_masked_sds: float = 0.0
        lambda_depth_masked_sds: float = 0.0
        mask_fill_method: str = "distance"
        mask_distance_max_radius: int = 24
        distance_sigma: float = 8.0
        lambda_mask_penetration: float = 0.0
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()


        #Reference taken from initial model
        #TODO use actual mesh for better results
        self.initial_gaussian_model_ref = None

        try:
            self.initial_gaussian_model_ref = copy.deepcopy(self.geometry)
            self.initial_gaussian_model_ref.eval()
            for name,param in self.initial_gaussian_model_ref.named_parameters():
                param.data = param.data.detach().clone()
                param.requires_grad = False

            for name, buffer in self.initial_gaussian_model_ref.named_buffers():
                if buffer is not None:
                    buffer.data = buffer.data.detach().clone()
            threestudio.info(f"Initialized reference renderer for silhouette loss.")

            threestudio.info(f"Model id: {id(self.geometry)}")
            threestudio.info(f"Model ref id: {id(self.initial_gaussian_model_ref)}")
        except Exception as e:
            self.initial_gaussian_model_ref = None
            threestudio.info(f"Could not create initial Gaussian reference: {e}")

        self.renderer.configure(
        geometry=self.geometry,
        material=self.material,
        background=self.background,
        ref_geometry=self.initial_gaussian_model_ref
        )
        #Some logging to see if parameters got passed correctly
        # Print all loss config parameters
        threestudio.info("Loss config parameters:")
        for key, value in self.cfg.loss.items():
            threestudio.info(f"  {key}: {value}")

        # Print all geometry config parameters
        threestudio.info("Geometry config parameters:")
        for key, value in self.cfg.geometry.items():
            threestudio.info(f"  {key}: {value}")

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]
        
        guidance_out = self.guidance(
            guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False, return_full_sds=True
        )

        # get raw SDS tensors (later for masked SDS)
        latents    = guidance_out.pop("sds_latents", None)
        timesteps  = guidance_out.pop("sds_t",        None)
        noise      = guidance_out.pop("sds_noise",    None)
        noise_pred = guidance_out.pop("sds_noise_pred", None)


        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.cfg.loss.get("lambda_sds", 0.0) > 0.0:
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_sds += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_sparsity"] > 0.0:
            # loss_sparsity = (out["comp_mask"] ** 2 + 0.01).sqrt().mean()
            # self.log("train/loss_sparsity", loss_sparsity)
            # loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            loss_sparsity = -(self.geometry.get_opacity - 0.5).pow(2).mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        # Outlier loss
        exp_out = self.cfg.loss.get("lambda_scale_outlier", 0.0)
        if exp_out > 0.0:
            scales = self.geometry.get_scaling
            exp_outlier_loss = per_dim_log_smooth_l1_loss(scales, exp_out, quantile=0.95)
            self.log("train/loss_scale_outlier", exp_outlier_loss)
            loss += exp_outlier_loss

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        if out.__contains__("comp_pred_normal"):
            loss_pred_normal = torch.nn.functional.mse_loss(
                out["comp_pred_normal"], out["comp_normal"].detach()
            )
            loss += loss_pred_normal








        # --- Chamfer loss (point cloud similarity) ---
        lambda_chamfer = self.cfg.loss.get("lambda_chamfer", 0.0)
        if self.global_step >= self.cfg.geometry.densify_from_iter and self.global_step <= self.cfg.geometry.densify_until_iter:
            lambda_chamfer = 0.0
        if lambda_chamfer > 0.0 and hasattr(self.geometry, "get_xyz") and batch.get("ref_pc") is not None:
            pred_pc = self.geometry.get_xyz.unsqueeze(0).repeat(batch["c2w"].shape[0], 1, 1)
            ref_pc = self.initial_gaussian_model_ref.get_xyz.unsqueeze(0).repeat(batch["c2w"].shape[0], 1, 1)
            chamfer = chamfer_loss_p3d(pred_pc, ref_pc, lambda_chamfer)
            self.log("train/loss_chamfer", chamfer)
            if self.global_step % 100 == 0:
                threestudio.info(f"Chamfer loss: {chamfer.item()} at step {self.global_step}")
            loss += chamfer


        # --- Alpha silhouette losses ---
        λ_sil = self.cfg.loss.get("lambda_alpha_silhouette_loss", 0.0)
        if λ_sil > 0 and out.get("ref_mask") is not None:
            loss_sil = alpha_silhouette_loss(
                out["comp_mask"],
                out["ref_mask"],
                weight=λ_sil,
                reduction='mean',
                visual_log_callback=self.save_image_grid,
                global_step=self.global_step
            )
            loss_sds += loss_sil
            self.log("train/loss_silhouette", loss_sil)


        # --- Depth silhouette loss ---
        λ_dsil = self.cfg.loss.get("lambda_depth_silhouette_loss", 0.0)
        if λ_dsil > 0 and out.get("ref_depth") is not None:
            loss_dsil = depth_silhouette_loss(
                out["comp_depth"],
                out["ref_depth"],
                weight=λ_dsil,
                reduction='mean',
                visual_log_callback=self.save_image_grid,
                global_step=self.global_step
            )
            loss_sds += loss_dsil
            self.log("train/loss_depth_silhouette", loss_dsil)



        # masked SDS alpha+depth with hole filling
        λ_alpha = self.cfg.loss.get("lambda_alpha_masked_sds", 0.0)
        λ_depth = self.cfg.loss.get("lambda_depth_masked_sds", 0.0)
        λ_masked_sds = self.cfg.loss.get("lambda_masked_sds", 0.0)
        if λ_masked_sds > 0:
            loss_m = masked_sds_loss(
                out,
                latents,
                noise,
                noise_pred,
                self.cfg.loss,
                λ_alpha,
                λ_depth,
                λ_masked_sds,
                visual_log_callback=self.save_image_grid,
                global_step=self.global_step
            )
            self.log('train/loss_masked_alpha_depth', loss_m)
            loss_sds += loss_m

        # --- Mask penetration loss ---
        lambda_mask_pen = self.cfg.loss.get("lambda_mask_penetration", 0.0)
        if lambda_mask_pen > 0.0 and batch.get("ref_mask") is not None:
            mask_pen = mask_penetration_loss(
                out,
                scales=self.geometry.get_scaling,
                weight=lambda_mask_pen,
                visual_log_callback=self.visual_log_callback,
                global_step=self.global_step,
            )
            self.log("train/loss_mask_penetration", mask_pen)
            loss_sds += mask_pen

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_") and value > 0:
                self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        if loss > 0:
            loss.backward()
        nn_utils.clip_grad_norm_(self.geometry.parameters(), 0.1)
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds}

    def grab(self,out,key, is_grayscale=False, data_range=False, camp=None):
        img = out[key].detach().squeeze()
        return {
            "type": "grayscale" if is_grayscale else "rgb",
            "img": img,
            "kwargs": {
                **({"data_format": "HWC"} if not is_grayscale else {"cmap": camp}),
                **({"data_range": (0,1)} if data_range else {"data_range": None}),
            },
        }
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # debug info
        # [INFO] out[ref_rgb]: shape=(1, 512, 512, 3)
        # [INFO] out[ref_viewspace_points]: type=<class 'list'>
        # [INFO] out[ref_visibility_filter]: type=<class 'list'>
        # [INFO] out[ref_radii]: type=<class 'list'>
        # [INFO] out[ref_normal]: shape=(1, 512, 512, 3)
        # [INFO] out[ref_depth]: shape=(1, 512, 512, 1)
        # [INFO] out[ref_mask]: shape=(1, 512, 512, 1)
        # [INFO] out[comp_rgb]: shape=(1, 512, 512, 3)
        # [INFO] out[viewspace_points]: type=<class 'list'>
        # [INFO] out[visibility_filter]: type=<class 'list'>
        # [INFO] out[radii]: type=<class 'list'>
        # [INFO] out[comp_normal]: shape=(1, 512, 512, 3)
        # [INFO] out[comp_depth]: shape=(1, 512, 512, 1)
        # [INFO] out[comp_mask]: shape=(1, 512, 512, 1)
        # [INFO] out[ref_rgb]: shape=(1, 512, 512, 3)
        # [INFO] out[ref_viewspace_points]: type=<class 'list'>
        # [INFO] out[ref_visibility_filter]: type=<class 'list'>
        # [INFO] out[ref_radii]: type=<class 'list'>
        # [INFO] out[ref_normal]: shape=(1, 512, 512, 3)
        # [INFO] out[ref_depth]: shape=(1, 512, 512, 1)
        # [INFO] out[ref_mask]: shape=(1, 512, 512, 1)

        # Print all geometry parameters and their shapes
        # [INFO] Geometry parameters and their shapes:
        # [INFO]   _xyz: (181690, 3)
        # [INFO]   _features_dc: (181690, 1, 3)
        # [INFO]   _features_rest: (181690, 0, 3)
        # [INFO]   _scaling: (181690, 3)
        # [INFO]   _rotation: (181690, 4)
        # [INFO]   _opacity: (181690, 1)
        images = [
            self.grab(out, "comp_rgb", data_range=True)
        ]


        # optional normals
        if "comp_normal" in out:
            images.append(self.grab(out,"comp_normal", data_range=True))
        if "comp_pred_normal" in out:
            images.append(self.grab(out,"comp_pred_normal", data_range=True))

        # masks
        if "comp_mask" in out:
            images.append(self.grab(out,"comp_mask", is_grayscale=True))
        if "ref_mask" in out:
            images.append(self.grab(out,"ref_mask", is_grayscale=True))

        # depth
        if "comp_depth" in out:
            images.append(self.grab(out,"comp_depth", is_grayscale=True, camp="jet"))
        if "ref_depth" in out:
            images.append(self.grab(out,"ref_depth", is_grayscale=True, camp="jet"))

        # now everything in `images` is guaranteed 3D
        self.save_image_grid(
            f"validation/it{self.global_step}-{batch['index'][0]}.png",
            images,
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        images = [
            self.grab(out, "comp_rgb", data_range=True),
        ]
        # save the image grid
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            images,
            name="test_step",
            step=self.global_step,
        )

        # save point cloud once
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
