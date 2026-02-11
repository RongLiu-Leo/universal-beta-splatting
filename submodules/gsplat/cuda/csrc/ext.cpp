#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def(
        "cond_mean_convariance_opacity_fwd",
        &gsplat::cond_mean_convariance_opacity_fwd_tensor
    );
    m.def(
        "cond_mean_convariance_opacity_bwd",
        &gsplat::cond_mean_convariance_opacity_bwd_tensor
    );

    m.def(
        "rot_scale_l_triangle_to_covar_fwd",
        &gsplat::rot_scale_l_triangle_to_covar_fwd_tensor
    );
    m.def(
        "rot_scale_l_triangle_to_covar_bwd",
        &gsplat::rot_scale_l_triangle_to_covar_bwd_tensor
    );

    m.def("l_triangle_to_rotmat_fwd", &gsplat::l_triangle_to_rotmat_fwd_tensor);
    m.def("l_triangle_to_rotmat_bwd", &gsplat::l_triangle_to_rotmat_bwd_tensor);

    m.def(
        "quat_scale_to_covar_preci_fwd",
        &gsplat::quat_scale_to_covar_preci_fwd_tensor
    );
    m.def(
        "quat_scale_to_covar_preci_bwd",
        &gsplat::quat_scale_to_covar_preci_bwd_tensor
    );

    m.def("proj_fwd", &gsplat::proj_fwd_tensor);
    m.def("proj_bwd", &gsplat::proj_bwd_tensor);

    m.def("world_to_cam_fwd", &gsplat::world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &gsplat::world_to_cam_bwd_tensor);

    m.def(
        "fully_fused_projection_fwd", &gsplat::fully_fused_projection_fwd_tensor
    );
    m.def(
        "fully_fused_projection_bwd", &gsplat::fully_fused_projection_bwd_tensor
    );

    m.def("isect_tiles", &gsplat::isect_tiles_tensor);
    m.def("isect_offset_encode", &gsplat::isect_offset_encode_tensor);

    m.def("rasterize_to_pixels_fwd", &gsplat::rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);

    m.def(
        "rasterize_to_indices_in_range",
        &gsplat::rasterize_to_indices_in_range_tensor
    );
}
