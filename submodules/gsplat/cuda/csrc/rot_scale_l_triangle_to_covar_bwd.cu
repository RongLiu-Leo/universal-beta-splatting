#include "bindings.h"
#include "helpers.cuh"
#include <cuda_runtime.h>

namespace gsplat {

template <typename T>
__global__ void rot_scale_l_triangle_to_covar_bwd_kernel(
    int64_t N,
    int64_t D,
    const T* __restrict__ rot,         // [N,3,3]
    const T* __restrict__ scale,       // [N,D]
    const T* __restrict__ l_triangle,  // [N, full_M], full_M = D*(D-1)/2; first 3 unused
    const int* __restrict__ rest_i,    // [M_rest]
    const int* __restrict__ rest_j,    // [M_rest]
    int64_t M_rest,
    bool spatial_block,
    const T* __restrict__ v_covar,     // [N,dim,dim] (dim=3 or D)
    T* __restrict__ v_rot,             // [N,3,3]
    T* __restrict__ v_scale,           // [N,D]
    T* __restrict__ v_l_triangle       // [N, full_M]
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Dimensions / pointers per item
    const int64_t full_M = D * (D - 1) / 2;
    const int dim = (spatial_block || D == 3) ? 3 : (int)D;

    const T* Rptr = rot   + idx * 9;          // 3x3
    const T* Sptr = scale + idx * D;          // D
    const T* Ltri = l_triangle + idx * full_M;   // full lower-tri storage
    const T* Gptr = v_covar + idx * dim * dim;   // upstream grad wrt C

    T* dR = v_rot + idx * 9;
    T* dS = v_scale + idx * D;
    T* dLT = v_l_triangle + idx * full_M;

    // Zero outputs (safe, N is typically large but D is small)
    // rot
    #pragma unroll
    for (int i = 0; i < 9; ++i) dR[i] = T(0);
    // scale
    for (int i = 0; i < D; ++i) dS[i] = T(0);
    // l_triangle
    for (int i = 0; i < full_M; ++i) dLT[i] = T(0);

    // Build Lxyz = R * diag(scale[:3])
    T Lxyz[9];
    Lxyz[0] = Rptr[0] * Sptr[0]; Lxyz[1] = Rptr[1] * Sptr[1]; Lxyz[2] = Rptr[2] * Sptr[2];
    Lxyz[3] = Rptr[3] * Sptr[0]; Lxyz[4] = Rptr[4] * Sptr[1]; Lxyz[5] = Rptr[5] * Sptr[2];
    Lxyz[6] = Rptr[6] * Sptr[0]; Lxyz[7] = Rptr[7] * Sptr[1]; Lxyz[8] = Rptr[8] * Sptr[2];

    // Build row-start/len for rest rows (>=3) similar to fwd
    const int MAX_D = 64; // supports up to D<=64
    int row_start[MAX_D];
    int row_len[MAX_D];
    #pragma unroll
    for (int r = 0; r < MAX_D; ++r) { row_start[r] = -1; row_len[r] = 0; }
    for (int m = 0; m < M_rest; ++m) {
        const int ri = rest_i[m];
        if (ri >= D) continue;
        if (row_start[ri] == -1) row_start[ri] = m;
        row_len[ri] += 1;
    }

    // Helper lambdas to fetch (G+G^T) entries without materializing S
    auto S_ij = [&](int i, int j) -> T {
        return Gptr[i * dim + j] + Gptr[j * dim + i];
    };

    // --- If only spatial block (3x3) ---
    if (spatial_block || D == 3) {
        // H = (G+G^T) * L_EE   where L_EE = Lxyz (3x3)
        T H[9];
        // First compute S_EE (3x3)
        T S00 = S_ij(0,0), S01 = S_ij(0,1), S02 = S_ij(0,2);
        T S10 = S_ij(1,0), S11 = S_ij(1,1), S12 = S_ij(1,2);
        T S20 = S_ij(2,0), S21 = S_ij(2,1), S22 = S_ij(2,2);

        // H = S_EE * Lxyz
        H[0] = S00*Lxyz[0] + S01*Lxyz[3] + S02*Lxyz[6];
        H[1] = S00*Lxyz[1] + S01*Lxyz[4] + S02*Lxyz[7];
        H[2] = S00*Lxyz[2] + S01*Lxyz[5] + S02*Lxyz[8];

        H[3] = S10*Lxyz[0] + S11*Lxyz[3] + S12*Lxyz[6];
        H[4] = S10*Lxyz[1] + S11*Lxyz[4] + S12*Lxyz[7];
        H[5] = S10*Lxyz[2] + S11*Lxyz[5] + S12*Lxyz[8];

        H[6] = S20*Lxyz[0] + S21*Lxyz[3] + S22*Lxyz[6];
        H[7] = S20*Lxyz[1] + S21*Lxyz[4] + S22*Lxyz[7];
        H[8] = S20*Lxyz[2] + S21*Lxyz[5] + S22*Lxyz[8];

        // dR_ij = H_ij * s_j ;  dS_j += sum_i H_ij * R_ij
        // j=0
        dR[0] = H[0] * Sptr[0]; dR[3] = H[3] * Sptr[0]; dR[6] = H[6] * Sptr[0];
        dS[0] = H[0]*Rptr[0] + H[3]*Rptr[3] + H[6]*Rptr[6];
        // j=1
        dR[1] = H[1] * Sptr[1]; dR[4] = H[4] * Sptr[1]; dR[7] = H[7] * Sptr[1];
        dS[1] = H[1]*Rptr[1] + H[4]*Rptr[4] + H[7]*Rptr[7];
        // j=2
        dR[2] = H[2] * Sptr[2]; dR[5] = H[5] * Sptr[2]; dR[8] = H[8] * Sptr[2];
        dS[2] = H[2]*Rptr[2] + H[5]*Rptr[5] + H[8]*Rptr[8];

        // l_triangle grads already zeroed (including first 3)
        return;
    }

    // --- Full D×D case ---
    // Precompute S_EE (3x3)
    T S00 = S_ij(0,0), S01 = S_ij(0,1), S02 = S_ij(0,2);
    T S10 = S_ij(1,0), S11 = S_ij(1,1), S12 = S_ij(1,2);
    T S20 = S_ij(2,0), S21 = S_ij(2,1), S22 = S_ij(2,2);

    // H = P_EE = S_EE*L_EE + S_EF*L_FE
    T H[9];
    // H <- S_EE * Lxyz
    H[0] = S00*Lxyz[0] + S01*Lxyz[3] + S02*Lxyz[6];
    H[1] = S00*Lxyz[1] + S01*Lxyz[4] + S02*Lxyz[7];
    H[2] = S00*Lxyz[2] + S01*Lxyz[5] + S02*Lxyz[8];

    H[3] = S10*Lxyz[0] + S11*Lxyz[3] + S12*Lxyz[6];
    H[4] = S10*Lxyz[1] + S11*Lxyz[4] + S12*Lxyz[7];
    H[5] = S10*Lxyz[2] + S11*Lxyz[5] + S12*Lxyz[8];

    H[6] = S20*Lxyz[0] + S21*Lxyz[3] + S22*Lxyz[6];
    H[7] = S20*Lxyz[1] + S21*Lxyz[4] + S22*Lxyz[7];
    H[8] = S20*Lxyz[2] + S21*Lxyz[5] + S22*Lxyz[8];

    // Add S_EF * L_FE:
    // For each row c in F, add outer product S[:,c] (rows i in E) with L_FE[c,:] (cols k in E)
    for (int c = 3; c < D; ++c) {
        const int rs = row_start[c];
        const int rl = row_len[c];
        if (rs == -1) continue;
        // Gather S(i,c) for i=0..2 once
        const T S0c = S_ij(0, c);
        const T S1c = S_ij(1, c);
        const T S2c = S_ij(2, c);
        for (int t = 0; t < rl; ++t) {
            const int m = rs + t;
            const int k = rest_j[m];
            if (k >= 3) continue; // only cross-to-spatial
            const T xck = Ltri[3 + m];
            // H[i,k] += S[i,c] * x_{c,k}
            if (k == 0) { H[0] += S0c * xck; H[3] += S1c * xck; H[6] += S2c * xck; }
            else if (k == 1) { H[1] += S0c * xck; H[4] += S1c * xck; H[7] += S2c * xck; }
            else /*k==2*/ { H[2] += S0c * xck; H[5] += S1c * xck; H[8] += S2c * xck; }
        }
    }

    // dR_ij = H_ij * s_j ;  dS_j (j<3) += sum_i H_ij * R_ij
    // j=0
    dR[0] = H[0] * Sptr[0]; dR[3] = H[3] * Sptr[0]; dR[6] = H[6] * Sptr[0];
    dS[0] = H[0]*Rptr[0] + H[3]*Rptr[3] + H[6]*Rptr[6];
    // j=1
    dR[1] = H[1] * Sptr[1]; dR[4] = H[4] * Sptr[1]; dR[7] = H[7] * Sptr[1];
    dS[1] = H[1]*Rptr[1] + H[4]*Rptr[4] + H[7]*Rptr[7];
    // j=2
    dR[2] = H[2] * Sptr[2]; dR[5] = H[5] * Sptr[2]; dR[8] = H[8] * Sptr[2];
    dS[2] = H[2]*Rptr[2] + H[5]*Rptr[5] + H[8]*Rptr[8];

    // --- Gradients for x_{r,c} (rest entries) and s_r (r>=3) ---

    // For convenience, define a tiny accessor to test if row j has a column q and fetch its value/grad index
    auto find_in_row = [&](int row, int col, T& val, int& m_idx, bool want_value) -> bool {
        const int rs = row_start[row];
        if (rs == -1) return false;
        const int rl = row_len[row];
        for (int tt = 0; tt < rl; ++tt) {
            const int mm = rs + tt;
            if (rest_j[mm] == col) {
                m_idx = mm;
                if (want_value) val = Ltri[3 + mm];
                return true;
            }
        }
        return false;
    };

    // 1) Off-diagonal and cross terms: for every stored (r,c) with r>=3, c<r
    for (int m = 0; m < M_rest; ++m) {
        const int r = rest_i[m];   // row
        const int c = rest_j[m];   // col (< r)
        T grad = T(0);

        if (c < 3) {
            // d/d x_{r,c} = (S_FE * L_EE + S_FF * L_FE)_{r-3, c}
            // = sum_{i=0..2} S[r,i] * L_EE[i,c] + sum_{j>=3} S[r,j] * x_{j,c}
            // First term: S[r,i] * Lxyz[i,c]
            grad += S_ij(r, 0) * Lxyz[0*3 + c];
            grad += S_ij(r, 1) * Lxyz[1*3 + c];
            grad += S_ij(r, 2) * Lxyz[2*3 + c];

            // Second term: over j>=3 where x_{j,c} exists
            for (int j = 3; j < D; ++j) {
                T xjc = T(0);
                int j_idx = -1;
                if (find_in_row(j, c, xjc, j_idx, /*want_value=*/true)) {
                    grad += S_ij(r, j) * xjc;
                }
            }
        } else {
            // c>=3, strictly-lower in L_FF
            // d/d x_{r,c} = (S_FF * L_FF)_{r,c}
            //            = S[r,c]*s_c + sum_{j=c+1..D-1} S[r,j]*x_{j,c}
            grad += S_ij(r, c) * Sptr[c];
            for (int j = c + 1; j < D; ++j) {
                T xjc = T(0);
                int j_idx = -1;
                if (find_in_row(j, c, xjc, j_idx, /*want_value=*/true)) {
                    grad += S_ij(r, j) * xjc;
                }
            }
        }
        dLT[3 + m] = grad; // gradient aligned with the same compact storage (+3 offset)
    }

    // 2) Diagonals in L_FF: s_r for r>=3
    for (int r = 3; r < D; ++r) {
        // ds_r = (S_FF * L_FF)_{r,r} = S[r,r]*s_r + sum_{j=r+1..D-1} S[r,j] * x_{j,r}
        T ds = S_ij(r, r) * Sptr[r];
        for (int j = r + 1; j < D; ++j) {
            T xjr = T(0);
            int j_idx = -1;
            if (find_in_row(j, r, xjr, j_idx, /*want_value=*/true)) {
                ds += S_ij(r, j) * xjr;
            }
        }
        dS[r] = ds;
    }

    // Ensure the first 3 entries of l_triangle grads are zero (already zeroed)
    // dLT[0..2] kept 0
}

// Binding to Python
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rot_scale_l_triangle_to_covar_bwd_tensor(
    const torch::Tensor& rot,          // [N,3,3]
    const torch::Tensor& scale,        // [N,D]
    const torch::Tensor& l_triangle,   // [N, full_M]
    const torch::Tensor& rest_i,       // [M_rest] int32
    const torch::Tensor& rest_j,       // [M_rest] int32
    bool spatial_block,
    const torch::Tensor& v_covar       // [N,dim,dim] (dim=3 or D)
) {
    GSPLAT_DEVICE_GUARD(rot);
    GSPLAT_CHECK_INPUT(rot);
    GSPLAT_CHECK_INPUT(scale);
    GSPLAT_CHECK_INPUT(l_triangle);
    GSPLAT_CHECK_INPUT(rest_i);
    GSPLAT_CHECK_INPUT(rest_j);
    GSPLAT_CHECK_INPUT(v_covar);

    const int64_t N = scale.size(0);
    const int64_t D = scale.size(1);
    const int64_t full_M = D * (D - 1) / 2;
    const int64_t M_rest = rest_i.numel();

    auto opts = rot.options();
    auto v_rot     = torch::zeros({N, 3, 3}, opts);
    auto v_scale   = torch::zeros({N, D}, opts);
    auto v_ltri    = torch::zeros({N, full_M}, opts);

    if (N > 0) {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;
        auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            rot.scalar_type(),
            "rot_scale_l_triangle_to_covar_bwd", [&]() {
                rot_scale_l_triangle_to_covar_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    N, D,
                    rot.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    l_triangle.data_ptr<scalar_t>(),
                    rest_i.data_ptr<int>(),
                    rest_j.data_ptr<int>(),
                    M_rest,
                    spatial_block,
                    v_covar.data_ptr<scalar_t>(),
                    v_rot.data_ptr<scalar_t>(),
                    v_scale.data_ptr<scalar_t>(),
                    v_ltri.data_ptr<scalar_t>()
                );
            }
        );
    }

    return {v_rot, v_scale, v_ltri};
}

} // namespace gsplat