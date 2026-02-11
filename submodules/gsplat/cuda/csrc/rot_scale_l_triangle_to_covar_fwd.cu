#include "bindings.h"
#include "helpers.cuh"
#include <cuda_runtime.h>

namespace gsplat {

template <typename T>
__global__ void rot_scale_l_triangle_to_covar_fwd_kernel(
    int64_t N,
    int64_t D,
    const T* __restrict__ rot,         // [N, 3, 3]
    const T* __restrict__ scale,       // [N, D]
    const T* __restrict__ l_triangle,  // [N, D*(D-1)/2]
    const int* __restrict__ rest_i,    // [M_rest]
    const int* __restrict__ rest_j,    // [M_rest]
    int64_t M_rest,
    bool spatial_block,
    T* __restrict__ covar              // [N, (spatial?3:D), (spatial?3:D)]
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ---- Pointers for this primitive ----
    const T* Rptr = rot   + idx * 9;          // 3x3
    const T* Sptr = scale + idx * D;          // D
    const T* x_rest = l_triangle + idx * ((D * (D - 1)) / 2) + 3;  // skip first 3

    // ---- L_xyz = R * diag(scale[:3]) ----
    // Column-wise scaling: Lxyz[i,j] = R[i,j] * s[j]
    T Lxyz[9];
    Lxyz[0] = Rptr[0] * Sptr[0]; Lxyz[1] = Rptr[1] * Sptr[1]; Lxyz[2] = Rptr[2] * Sptr[2];
    Lxyz[3] = Rptr[3] * Sptr[0]; Lxyz[4] = Rptr[4] * Sptr[1]; Lxyz[5] = Rptr[5] * Sptr[2];
    Lxyz[6] = Rptr[6] * Sptr[0]; Lxyz[7] = Rptr[7] * Sptr[1]; Lxyz[8] = Rptr[8] * Sptr[2];

    // ---- Fast path: only 3x3 spatial block ----
    if (spatial_block || D == 3) {
        T* C = covar + idx * 9;
        // C = Lxyz * Lxyz^T
        const T c00 = Lxyz[0]*Lxyz[0] + Lxyz[1]*Lxyz[1] + Lxyz[2]*Lxyz[2];
        const T c01 = Lxyz[0]*Lxyz[3] + Lxyz[1]*Lxyz[4] + Lxyz[2]*Lxyz[5];
        const T c02 = Lxyz[0]*Lxyz[6] + Lxyz[1]*Lxyz[7] + Lxyz[2]*Lxyz[8];
        const T c11 = Lxyz[3]*Lxyz[3] + Lxyz[4]*Lxyz[4] + Lxyz[5]*Lxyz[5];
        const T c12 = Lxyz[3]*Lxyz[6] + Lxyz[4]*Lxyz[7] + Lxyz[5]*Lxyz[8];
        const T c22 = Lxyz[6]*Lxyz[6] + Lxyz[7]*Lxyz[7] + Lxyz[8]*Lxyz[8];

        C[0] = c00; C[1] = c01; C[2] = c02;
        C[3] = c01; C[4] = c11; C[5] = c12;
        C[6] = c02; C[7] = c12; C[8] = c22;
        return;
    }

    // ---- Full D×D covariance ----
    T* Cfull = covar + idx * D * D;

    // 1) Top-left 3×3 block: C00 = Lxyz * Lxyz^T
    {
        const T c00 = Lxyz[0]*Lxyz[0] + Lxyz[1]*Lxyz[1] + Lxyz[2]*Lxyz[2];
        const T c01 = Lxyz[0]*Lxyz[3] + Lxyz[1]*Lxyz[4] + Lxyz[2]*Lxyz[5];
        const T c02 = Lxyz[0]*Lxyz[6] + Lxyz[1]*Lxyz[7] + Lxyz[2]*Lxyz[8];
        const T c11 = Lxyz[3]*Lxyz[3] + Lxyz[4]*Lxyz[4] + Lxyz[5]*Lxyz[5];
        const T c12 = Lxyz[3]*Lxyz[6] + Lxyz[4]*Lxyz[7] + Lxyz[5]*Lxyz[8];
        const T c22 = Lxyz[6]*Lxyz[6] + Lxyz[7]*Lxyz[7] + Lxyz[8]*Lxyz[8];

        Cfull[0*D + 0] = c00; Cfull[0*D + 1] = c01; Cfull[0*D + 2] = c02;
        Cfull[1*D + 0] = c01; Cfull[1*D + 1] = c11; Cfull[1*D + 2] = c12;
        Cfull[2*D + 0] = c02; Cfull[2*D + 1] = c12; Cfull[2*D + 2] = c22;
    }

    // 2) Build compact row ranges for the "rest" lower-tri entries (rows >= 3).
    // We only need tiny per-thread metadata: start index and length per row.
    // Assumes rest_i is grouped by row (common in lower-tri generation).
    // D is typically small (<= 10 in your tests), so these small arrays are fine.
    const int MAX_D = 64; // supports comfortably up to D=64
    int row_start[64];
    int row_len[64];

    // init
    #pragma unroll
    for (int r = 0; r < 64; ++r) { row_start[r] = -1; row_len[r] = 0; }

    // single pass to record row starts and counts
    for (int m = 0; m < M_rest; ++m) {
        const int ri = rest_i[m];
        if (ri >= D) continue;  // safety
        if (row_start[ri] == -1) row_start[ri] = m;
        row_len[ri] += 1;
    }

    // 3) Cross block C[0:3, 3:D] and its transpose:
    //    For each col c>=3, only the entries with (row=c, col<3) contribute with Lxyz.
    for (int c = 3; c < D; ++c) {
        T s0 = T(0), s1 = T(0), s2 = T(0);
        const int rs = row_start[c];
        const int rl = row_len[c];
        if (rs != -1) {
            for (int t = 0; t < rl; ++t) {
                const int m = rs + t;
                const int k = rest_j[m];
                if (k < 3) {
                    const T v = x_rest[m];
                    s0 += Lxyz[0*3 + k] * v;
                    s1 += Lxyz[1*3 + k] * v;
                    s2 += Lxyz[2*3 + k] * v;
                }
            }
        }
        // write C[0..2, c] and mirror
        Cfull[0*D + c] = s0; Cfull[c*D + 0] = s0;
        Cfull[1*D + c] = s1; Cfull[c*D + 1] = s1;
        Cfull[2*D + c] = s2; Cfull[c*D + 2] = s2;
    }

    // 4) Diagonal for rows >=3:
    for (int r = 3; r < D; ++r) {
        T sumsq = Sptr[r] * Sptr[r];  // diag contribution
        const int rs = row_start[r];
        const int rl = row_len[r];
        if (rs != -1) {
            for (int t = 0; t < rl; ++t) {
                const T v = x_rest[rs + t];
                sumsq += v * v;
            }
        }
        Cfull[r*D + r] = sumsq;
    }

    // 5) Off-diagonals for rows/cols >=3, fill lower triangle and mirror.
    for (int r = 4; r < D; ++r) {
        const int rs_r = row_start[r];
        const int rl_r = row_len[r];

        for (int c = 3; c < r; ++c) {
            const int rs_c = row_start[c];
            const int rl_c = row_len[c];

            // dot over k < c: sum_{k < c} L[r,k] * L[c,k]
            // Intersect the two (row-wise) lists; both are tiny (<= r, <= c).
            T s = T(0);
            if (rs_r != -1 && rs_c != -1) {
                // iterate over the shorter row to reduce work
                if (rl_r <= rl_c) {
                    for (int tr = 0; tr < rl_r; ++tr) {
                        const int m_r = rs_r + tr;
                        const int k   = rest_j[m_r];
                        if (k >= c) continue;  // only k < c contribute to the shared sum
                        const T vrk = x_rest[m_r];

                        // search for same k in row c
                        T vck = T(0);
                        for (int tc = 0; tc < rl_c; ++tc) {
                            const int m_c = rs_c + tc;
                            if (rest_j[m_c] == k) { vck = x_rest[m_c]; break; }
                        }
                        s += vrk * vck;
                    }
                } else {
                    for (int tc = 0; tc < rl_c; ++tc) {
                        const int m_c = rs_c + tc;
                        const int k   = rest_j[m_c];
                        if (k >= c) continue;
                        const T vck = x_rest[m_c];

                        T vrk = T(0);
                        for (int tr = 0; tr < rl_r; ++tr) {
                            const int m_r = rs_r + tr;
                            if (rest_j[m_r] == k) { vrk = x_rest[m_r]; break; }
                        }
                        s += vrk * vck;
                    }
                }
            }

            // plus s[c] * x[r,c] term if that entry exists in row r
            T x_rc = T(0);
            if (rs_r != -1) {
                for (int tr = 0; tr < rl_r; ++tr) {
                    const int m_r = rs_r + tr;
                    if (rest_j[m_r] == c) { x_rc = x_rest[m_r]; break; }
                }
            }
            s += Sptr[c] * x_rc;

            // write symmetric pair
            Cfull[r*D + c] = s;
            Cfull[c*D + r] = s;
        }
    }
}

// Binding to Python
torch::Tensor rot_scale_l_triangle_to_covar_fwd_tensor(
    const torch::Tensor& rot,
    const torch::Tensor& scale,
    const torch::Tensor& l_triangle,
    const torch::Tensor& rest_i,
    const torch::Tensor& rest_j,
    bool spatial_block
) {
    GSPLAT_DEVICE_GUARD(rot);
    GSPLAT_CHECK_INPUT(rot);
    GSPLAT_CHECK_INPUT(scale);
    GSPLAT_CHECK_INPUT(l_triangle);
    GSPLAT_CHECK_INPUT(rest_i);
    GSPLAT_CHECK_INPUT(rest_j);

    const int64_t N = scale.size(0);
    const int64_t D = scale.size(1);
    const int64_t M_rest = rest_i.numel();

    auto covar = torch::empty({N, spatial_block ? 3 : D, spatial_block ? 3 : D}, rot.options());

    if (N > 0) {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            rot.scalar_type(),
            "rot_scale_l_triangle_to_covar_fwd", [&]() {
                rot_scale_l_triangle_to_covar_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    N, D,
                    rot.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    l_triangle.data_ptr<scalar_t>(),
                    rest_i.data_ptr<int>(),
                    rest_j.data_ptr<int>(),
                    M_rest,
                    spatial_block,
                    covar.data_ptr<scalar_t>()
                );
            }
        );
    }
    return covar;
}

} // namespace gsplat
