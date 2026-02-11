import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, apply_depth_colormap
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.cuda._torch_impl import (
    _l_triangle_to_rotmat,
    _rot_scale_l_triangle_to_covar,
    _cond_mean_convariance_opacity,
)
from gsplat.cuda._wrapper import (
    l_triangle_to_rotmat,
    rot_scale_l_triangle_to_covar,
    cond_mean_convariance_opacity,
)
import json
import time
from .beta_viewer import BetaRenderTabState


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class BetaModel:
    def setup_functions(self):
        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        def inverse_softplus(y):
            return y + torch.log(-torch.expm1(-y))

        self.scale_activation = F.softplus
        self.scale_inverse_activation = inverse_softplus

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.beta_activation = beta_activation

        self.l_triangs_activation = lambda x: x
        self.l_triangs_inverse_activation = lambda x: x

    def __init__(self, input_dim: int = 6):
        self.input_dim = input_dim

        self._xyz = torch.empty(0)
        self._mean = torch.empty(0)
        self._scale = torch.empty(0)
        self._l_triangle = torch.empty(0)
        self._rgb = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.background = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()

        tril_i, tril_j = torch.tril_indices(input_dim, input_dim, offset=-1)
        # mask out the first 3 skew params (used in get_rotation)
        mask_rest = (tril_i >= 3) | (tril_j >= 3)
        self.rest_i = tril_i[mask_rest].to(torch.int32).to("cuda")
        self.rest_j = tril_j[mask_rest].to(torch.int32).to("cuda")

    def capture(self):
        return (
            self._xyz,
            self._mean,
            self._scale,
            self._l_triangle,
            self._rgb,
            self._opacity,
            self._beta,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._mean,
            self._scale,
            self._l_triangle,
            self._rgb,
            self._opacity,
            self._beta,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scale(self):
        return self.scale_activation(self._scale)

    @property
    def get_l_triangle(self):
        return self.l_triangs_activation(self._l_triangle)

    @property
    def get_mean(self):
        return torch.cat([self._xyz, self._mean], dim=-1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_beta(self):
        return self.beta_activation(self._beta)

    @property
    def get_rotation(self):
        return l_triangle_to_rotmat(self.get_l_triangle[:, :3])

    # @property
    # def get_covariance(self):
    #     d = self.get_scale
    #     R = self.get_rotation
    #     return torch.einsum("nik,nk,njk->nij", R, d**2, R)

    @property
    def get_covariance(self):
        return rot_scale_l_triangle_to_covar(
            self.get_rotation,
            self.get_scale,
            self.get_l_triangle,
            self.rest_i,
            self.rest_j,
        )

    @property
    def get_xyz_covariance(self):
        return rot_scale_l_triangle_to_covar(
            self.get_rotation,
            self.get_scale,
            self.get_l_triangle,
            self.rest_i,
            self.rest_j,
            spatial_block=True,
        )

    def get_cond_mean_convariance_opacity(self, q):
        v = self.get_covariance
        m = self.get_mean
        o = self.get_opacity
        b = self.get_beta[:, 1:]
        return cond_mean_convariance_opacity(m, v, o, b, q)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        xyzs = fused_point_cloud
        means = torch.empty(fused_point_cloud.shape[0], 3, device="cuda").uniform_(
            -1.0, 1.0
        )
        if self.input_dim == 7:
            means_time = torch.empty(
                fused_point_cloud.shape[0], 1, device="cuda"
            ).uniform_(0.0, 1.0)
            means = torch.cat([means, means_time], dim=1)

        dist2 = (
            knn(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[:, 1:] ** 2
        ).mean(dim=-1)

        scales = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(
            1, 3
        )
        if self.input_dim > 3:
            scales_rest = self.scale_inverse_activation(
                torch.normal(
                    1,
                    1e-5,
                    size=(fused_point_cloud.shape[0], self.input_dim - 3),
                    device="cuda",
                )
            )
            scales = torch.cat([scales, scales_rest], dim=1)

        l_triangles = self.l_triangs_inverse_activation(
            torch.normal(
                0,
                1e-5,
                size=(
                    fused_point_cloud.shape[0],
                    (self.input_dim**2 + self.input_dim) // 2 - self.input_dim,
                ),
                device="cuda",
            )
        )

        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        betas = torch.zeros(
            (fused_point_cloud.shape[0], self.input_dim - 2),
            dtype=torch.float,
            device="cuda",
        )
        if self.input_dim == 7:
            betas[:, 1:4] -= 3

        self._xyz = nn.Parameter(xyzs.requires_grad_(True))
        self._mean = nn.Parameter(means.requires_grad_(True))
        self._rgb = nn.Parameter(fused_color.requires_grad_(True))
        self._scale = nn.Parameter(scales.requires_grad_(True))
        self._l_triangle = nn.Parameter(l_triangles.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))

    def prune(self, live_mask):
        self._xyz = self._xyz[live_mask]
        self._mean = self._mean[live_mask]
        self._rgb = self._rgb[live_mask]
        self._scale = self._scale[live_mask]
        self._l_triangle = self._l_triangle[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]

    def training_setup(self, training_args):
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._mean],
                "lr": training_args.mean_lr,
                "name": "mean",
            },
            {"params": [self._rgb], "lr": training_args.rgb_lr, "name": "rgb"},
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {"params": [self._beta], "lr": training_args.beta_lr, "name": "beta"},
            {
                "params": [self._scale],
                "lr": training_args.scale_lr,
                "name": "scale",
            },
            {
                "params": [self._l_triangle],
                "lr": training_args.l_triangle_lr,
                "name": "l_triangle",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "red", "green", "blue", "opacity"]
        for i in range(self._beta.shape[1]):
            l.append("beta_{}".format(i))
        for i in range(self.input_dim - 3):
            l.append("mean_{}".format(i))
        for i in range(self._scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._l_triangle.shape[1]):
            l.append("l_triangle_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        mean = self._mean.detach().cpu().numpy()
        rgb = self._rgb.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        l_triangle = self._l_triangle.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, rgb, opacities, betas, mean, scale, l_triangle),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        size_bytes = os.path.getsize(path) / (1024.0 * 1024.0)
        print(f"Loaded PLY size: {size_bytes} MB")

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        print(f"Loaded primitive number: {xyz.shape[0]}")
        mean_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("mean_")
        ]
        mean_names = sorted(mean_names, key=lambda x: int(x.split("_")[-1]))
        mean = np.zeros((xyz.shape[0], len(mean_names)))
        for idx, attr_name in enumerate(mean_names):
            mean[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        rgb = np.zeros((xyz.shape[0], 3))
        rgb[:, 0] = np.asarray(plydata.elements[0]["red"])
        rgb[:, 1] = np.asarray(plydata.elements[0]["green"])
        rgb[:, 2] = np.asarray(plydata.elements[0]["blue"])

        beta_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("beta_")
        ]
        beta_names = sorted(beta_names, key=lambda x: int(x.split("_")[-1]))
        betas = np.zeros((xyz.shape[0], len(beta_names)))
        for idx, attr_name in enumerate(beta_names):
            betas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        l_triangle_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("l_triangle")
        ]
        l_triangle_names = sorted(l_triangle_names, key=lambda x: int(x.split("_")[-1]))
        l_triangles = np.zeros((xyz.shape[0], len(l_triangle_names)))
        for idx, attr_name in enumerate(l_triangle_names):
            l_triangles[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._mean = nn.Parameter(
            torch.tensor(mean, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rgb = nn.Parameter(
            torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._beta = nn.Parameter(
            torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scale = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._l_triangle = nn.Parameter(
            torch.tensor(l_triangles, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_mean,
        new_rgb,
        new_opacities,
        new_betas,
        new_scale,
        new_l_triangle,
    ):
        d = {
            "xyz": new_xyz,
            "mean": new_mean,
            "rgb": new_rgb,
            "opacity": new_opacities,
            "beta": new_betas,
            "scale": new_scale,
            "l_triangle": new_l_triangle,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "mean": self._mean,
            "rgb": self._rgb,
            "opacity": self._opacity,
            "beta": self._beta,
            "scale": self._scale,
            "l_triangle": self._l_triangle,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]

            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._mean = optimizable_tensors["mean"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scale = optimizable_tensors["scale"]
        self._l_triangle = optimizable_tensors["l_triangle"]

        torch.cuda.empty_cache()

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(
            1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1)
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps,
            min=0.005,
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        return (
            self._xyz[idxs],
            self._mean[idxs],
            self._rgb[idxs],
            new_opacity,
            self._beta[idxs],
            self._scale[idxs],
            self._l_triangle[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        # print(f"Relocate: {dead_mask.sum().item()}")
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            relocated_xyz,
            relocated_mean,
            relocated_rgb,
            relocated_opacity,
            relocated_beta,
            relocated_scale,
            relocated_l_triangle,
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._xyz.index_copy_(0, dead_indices, relocated_xyz)
        self._mean.index_copy_(0, dead_indices, relocated_mean)
        self._rgb.index_copy_(0, dead_indices, relocated_rgb)
        self._opacity.index_copy_(0, dead_indices, relocated_opacity)
        self._beta.index_copy_(0, dead_indices, relocated_beta)
        self._scale.index_copy_(0, dead_indices, relocated_scale)
        self._l_triangle.index_copy_(0, dead_indices, relocated_l_triangle)

        self._opacity.index_copy_(
            0, reinit_idx, self._opacity.index_select(0, dead_indices)
        )

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.02 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        # print(f"Add: {num_gs}, Now {target_num}")

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_mean,
            new_rgb,
            new_opacity,
            new_beta,
            new_scale,
            new_l_triangle,
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity

        self.densification_postfix(
            new_xyz,
            new_mean,
            new_rgb,
            new_opacity,
            new_beta,
            new_scale,
            new_l_triangle,
        )
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def render(self, viewpoint_camera, render_mode="RGB", mask=None):
        if mask == None:
            mask = torch.ones_like(self.get_opacity.squeeze()).bool()

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)

        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)

        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        if self.input_dim > 3:
            cam_pos = viewpoint_camera.camera_center
            view_dir = self._xyz - cam_pos.unsqueeze(0)
            view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
            if self.input_dim == 6:
                query = view_dir
            elif self.input_dim == 7:
                timestamp = torch.full(
                    (view_dir.shape[0], 1),
                    viewpoint_camera.timestamp,
                    device=view_dir.device,
                    dtype=view_dir.dtype,
                )
                query = torch.cat([view_dir, timestamp], dim=-1)
            else:
                raise NotImplementedError("Only implemented for 6D or 7D query")
            means, convs, opacities = self.get_cond_mean_convariance_opacity(query)
        else:
            means = self.get_mean
            convs = self.get_covariance
            opacities = self.get_opacity

        rgbs, alphas, meta = rasterization(
            means=means[mask],
            l_triagnles=self.get_l_triangle[mask],
            scales=self.get_scale[mask],
            opacities=opacities.squeeze()[mask],
            betas=self.get_beta[:, :1].squeeze()[mask],
            colors=self._rgb[mask],
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode,
            covars=convs[mask],
        )

        # # Convert from N,H,W,C to N,C,H,W format
        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        return {
            "render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0,
        }

    @torch.no_grad()
    def view(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer."""
        assert isinstance(render_tab_state, BetaRenderTabState)

        def quantile_mask(beta, b_xyz=(0, 100), b_view=(0, 100), b_time=(0, 100)):
            """
            beta: [N, 2] -> (x, v) or [N, 3+] -> (x, v, t, ...)
            b_xyz, b_view, b_time: (lo%, hi%) in [0, 100]
            """
            qx_lo, qx_hi = b_xyz[0] / 100, b_xyz[1] / 100
            qv_lo, qv_hi = b_view[0] / 100, b_view[1] / 100

            x = beta[:, 0]
            v = beta[:, 1:4].mean(dim=-1)

            mask = (
                (x >= x.quantile(qx_lo))
                & (x <= x.quantile(qx_hi))
                & (v >= v.quantile(qv_lo))
                & (v <= v.quantile(qv_hi))
            )

            # Optional t-channel (if present)
            if b_time is not None:
                qt_lo, qt_hi = b_time[0] / 100, b_time[1] / 100
                t = beta[:, 4]
                mask = mask & (t >= t.quantile(qt_lo)) & (t <= t.quantile(qt_hi))

            return mask

        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((W, H))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")

        if center:
            self._xyz -= self._xyz.mean(dim=0, keepdim=True)

        if self.input_dim > 3:
            cam_pos = c2w[:3, 3]
            view_dir = self._xyz - cam_pos.unsqueeze(0)
            view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
            if self.input_dim == 6:
                query = view_dir
            elif self.input_dim == 7:
                timestamp = torch.full(
                    (view_dir.shape[0], 1),
                    render_tab_state.timestamp,
                    device=view_dir.device,
                    dtype=view_dir.dtype,
                )
                query = torch.cat([view_dir, timestamp], dim=-1)
            else:
                raise NotImplementedError("Only implemented for 6D or 7D query")
            means, convs, opacities = self.get_cond_mean_convariance_opacity(query)
        else:
            means = self.get_mean
            convs = self.get_covariance
            opacities = self.get_opacity

        render_mode = render_tab_state.render_mode

        mask = quantile_mask(
            self._beta,
            b_xyz=render_tab_state.b_xyz,
            b_view=render_tab_state.b_view,
            b_time=render_tab_state.b_time if self.input_dim == 7 else None,
        )

        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        render_colors, alphas, meta = rasterization(
            means=means[mask],
            l_triagnles=self.get_l_triangle[mask],
            scales=self.get_scale[mask],
            opacities=opacities.squeeze()[mask],
            betas=self.get_beta[:, :1].squeeze()[mask],
            colors=self._rgb[mask],
            viewmats=torch.linalg.inv(c2w).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode if render_mode != "Alpha" else "RGB",
            covars=convs[mask],
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
        )
        render_tab_state.total_count_number = len(self.get_mean)
        render_tab_state.rendered_count_number = (meta["radii"] > 0).sum().item()

        if render_mode == "Alpha":
            render_colors = alphas

        if render_colors.shape[-1] == 1:
            render_colors = apply_depth_colormap(render_colors)

        return render_colors[0].cpu().numpy()
