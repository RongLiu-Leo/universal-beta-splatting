import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
from .utils import depth_to_normal, get_projection_matrix


def rasterization(
    means: Tensor,  # [N, 3]
    l_triagnles: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    betas: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal[
        "RGB", "Normal", "Diffuse", "Specular", "Depth", "EDepth", "RGB+D", "RGB+ED"
    ] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:

    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert l_triagnles.shape == (N, 4), l_triagnles.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        l_triagnles, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert betas.shape == (N,), betas.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in [
        "RGB",
        "Normal",
        "Diffuse",
        "Specular",
        "Depth",
        "EDepth",
        "RGB+D",
        "RGB+ED",
    ], render_mode

    # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
    assert (colors.dim() == 2 and colors.shape[0] == N) or (
        colors.dim() == 3 and colors.shape[:2] == (C, N)
    ), colors.shape

    camtoworlds = torch.inverse(viewmats)
    # dirs = means[primitive_ids, :] - camtoworlds[camera_ids, :3, 3]

    # Project Gaussians to 2D. Directly pass in {l_triagnles, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        l_triagnles,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=(rasterize_mode == "antialiased"),
        ortho=False,
    )

    radii, means2d, depths, conics, compensations = proj_results
    opacities = opacities.repeat(C, 1)  # [C, N]
    betas = betas.repeat(C, 1)  # [C, N]
    camera_ids, primitive_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local primitive_ids
            "primitive_ids": primitive_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "betas": betas,
        }
    )

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if colors.dim() == 2:
        # Turn [N, D] into [C, N, D]
        colors = colors.expand(C, -1, -1)
    else:
        # colors is already [C, N, D]
        pass

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["Depth", "EDepth", "Normal"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        n_cameras=C,
        camera_ids=camera_ids,
        primitive_ids=primitive_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                betas,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode == "Normal":
        render_colors = depth_to_normal(render_colors, camtoworlds, Ks)
        render_colors = (render_colors + 1) / 2

    return render_colors, render_alphas, meta
