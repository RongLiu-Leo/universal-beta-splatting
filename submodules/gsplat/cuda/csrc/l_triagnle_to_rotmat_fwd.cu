#include "bindings.h"
#include "helpers.cuh"
#include <cuda_runtime.h>

namespace gsplat {

template <typename T>
__global__ void l_triangle_to_rotmat_fwd_kernel(
    const int64_t N,
    const T* __restrict__ l,   // [N, 3]
    T* __restrict__ R)         // [N, 3, 3]
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load skew entries
    const T a0 = l[idx * 3 + 0];  // A[0,1]
    const T a1 = l[idx * 3 + 1];  // A[0,2]
    const T a2 = l[idx * 3 + 2];  // A[1,2]

    // Pointer to output R (row-major)
    T* Rptr = R + idx * 9;

    // R = I + A
    Rptr[0] = T(1);
    Rptr[1] = a0;
    Rptr[2] = a1;

    Rptr[3] = -a0;
    Rptr[4] = T(1);
    Rptr[5] = a2;

    Rptr[6] = -a1;
    Rptr[7] = -a2;
    Rptr[8] = T(1);
}

torch::Tensor l_triangle_to_rotmat_fwd_tensor(
    const torch::Tensor& l_triangle
) {
    GSPLAT_DEVICE_GUARD(l_triangle);
    GSPLAT_CHECK_INPUT(l_triangle);
    TORCH_CHECK(l_triangle.size(1) == 3, "Expected l_triangle size[1]==3");

    int64_t N = l_triangle.size(0);
    auto R = torch::empty({N, 3, 3}, l_triangle.options());

    if (N > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            l_triangle.scalar_type(),
            "l_triangle_to_rotmat_fwd", [&]() {
                l_triangle_to_rotmat_fwd_kernel<scalar_t>
                    <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        N,
                        l_triangle.data_ptr<scalar_t>(),
                        R.data_ptr<scalar_t>()
                    );
            }
        );
    }
    return R;
}

} // namespace gsplat