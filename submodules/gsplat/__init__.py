import warnings

from .cuda._torch_impl import accumulate
from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    proj,
    quat_scale_to_covar_preci,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
    world_to_cam,
    l_triangle_to_rotmat,
    rot_scale_l_triangle_to_covar,
    cond_mean_convariance_opacity,
)
from .rendering import (
    rasterization,
)
from .version import __version__

all = [
    "rasterization",
    "rasterization_inria_wrapper",
    "isect_offset_encode",
    "isect_tiles",
    "proj",
    "fully_fused_projection",
    "quat_scale_to_covar_preci",
    "rasterize_to_pixels",
    "world_to_cam",
    "accumulate",
    "rasterize_to_indices_in_range",
    "l_triangle_to_rotmat",
    "rot_scale_l_triangle_to_covar",
    "cond_mean_convariance_opacity",
]
