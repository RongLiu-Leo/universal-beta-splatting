#include "bindings.h"
#include "helpers.cuh"
#include <cuda_runtime.h>

namespace gsplat {

template <typename T>
__global__ void l_triangle_to_rotmat_bwd_kernel(
    const int64_t N,
    const T* __restrict__ gradR, // [N, 3, 3]
    T* __restrict__ gradl)       // [N, 3]
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const T* Rg = gradR + idx * 9;
    T* lg = gradl + idx * 3;

    // dL0 = dR[0,1]*1 + dR[1,0]*(-1)
    lg[0] = Rg[1] - Rg[3];
    // dL1 = dR[0,2]*1 + dR[2,0]*(-1)
    lg[1] = Rg[2] - Rg[6];
    // dL2 = dR[1,2]*1 + dR[2,1]*(-1)
    lg[2] = Rg[5] - Rg[7];
}

torch::Tensor l_triangle_to_rotmat_bwd_tensor(
    const torch::Tensor& l_triangle,
    const torch::Tensor& gradR
) {
    GSPLAT_DEVICE_GUARD(l_triangle);
    GSPLAT_CHECK_INPUT(gradR);
    TORCH_CHECK(gradR.size(1) == 3 && gradR.size(2) == 3,
                "Expected gradR shape [N,3,3]");

    int64_t N = l_triangle.size(0);
    auto gradl = torch::empty_like(l_triangle);

    if (N > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            l_triangle.scalar_type(),
            "l_triangle_to_rotmat_bwd", [&]() {
                l_triangle_to_rotmat_bwd_kernel<scalar_t>
                <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                    N,
                    gradR.data_ptr<scalar_t>(),
                    gradl.data_ptr<scalar_t>()
                );
            }
        );
    }
    return gradl;
}

} // namespace gsplat
