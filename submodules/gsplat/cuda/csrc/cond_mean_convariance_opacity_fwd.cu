#include "bindings.h"
#include "helpers.cuh"
#include <cuda_runtime.h>
#include <limits>
#include <cmath>

namespace gsplat {

template <typename T>
__device__ inline T clamp01mEps(T x) {
    const T one = T(1);
    const T maxv = one - T(std::numeric_limits<T>::epsilon());
    if (x < T(0)) x = T(0);
    if (x > maxv) x = maxv;
    return x;
}

// Float-promoted math (safe for Half/BF16)
template <typename T> __device__ inline T fexp(T x)  { return T(expf(float(x))); }
template <typename T> __device__ inline T fpow(T a, T b)  { return T(powf(float(a), float(b))); }
template <typename T> __device__ inline T fsqrt(T x) { return T(sqrtf(float(x))); }
template <typename T> __device__ inline T ftanh(T x) { return T(tanhf(float(x))); }

// Small dense Gauss–Jordan inverse with partial pivoting.
// Works up to MAX_N; we only touch the upper-left n×n region.
template <int MAX_N, typename T>
__device__ inline void invert_small_matrix(const T* __restrict__ A, T* __restrict__ invA, int n) {
    T aug[(MAX_N) * (2 * MAX_N)];
#pragma unroll
    for (int r = 0; r < MAX_N; ++r) {
#pragma unroll
        for (int c = 0; c < 2 * MAX_N; ++c) aug[r * (2 * MAX_N) + c] = T(0);
    }
#pragma unroll
    for (int r = 0; r < n; ++r) {
#pragma unroll
        for (int c = 0; c < n; ++c) aug[r * (2 * MAX_N) + c] = A[r * n + c];
        aug[r * (2 * MAX_N) + (n + r)] = T(1);
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        T maxabs = fabsf(float(aug[piv * (2 * MAX_N) + col]));
        for (int r = col + 1; r < n; ++r) {
            T v = fabsf(float(aug[r * (2 * MAX_N) + col]));
            if (v > maxabs) { maxabs = v; piv = r; }
        }
        if (piv != col) {
#pragma unroll
            for (int c = 0; c < 2 * n; ++c) {
                T tmp = aug[col * (2 * MAX_N) + c];
                aug[col * (2 * MAX_N) + c] = aug[piv * (2 * MAX_N) + c];
                aug[piv * (2 * MAX_N) + c] = tmp;
            }
        }
        T diag = aug[col * (2 * MAX_N) + col];
        if (diag == T(0)) { const T eps = T(1e-20f); diag = (diag >= T(0)) ? eps : -eps; }
        T invdiag = T(1) / diag;
#pragma unroll
        for (int c = 0; c < 2 * n; ++c) aug[col * (2 * MAX_N) + c] *= invdiag;
        for (int r = 0; r < n; ++r) if (r != col) {
            T f = aug[r * (2 * MAX_N) + col];
            if (f != T(0)) {
#pragma unroll
                for (int c = 0; c < 2 * n; ++c) aug[r * (2 * MAX_N) + c] -= f * aug[col * (2 * MAX_N) + c];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < n; ++r) {
#pragma unroll
        for (int c = 0; c < n; ++c) invA[r * n + c] = aug[r * (2 * MAX_N) + (n + c)];
    }
}

template <typename T>
__device__ inline void invert_3x3(
    const T a00, const T a01, const T a02,
    const T a10, const T a11, const T a12,
    const T a20, const T a21, const T a22,
    T& i00, T& i01, T& i02,
    T& i10, T& i11, T& i12,
    T& i20, T& i21, T& i22)
{
    const T c00 = a11 * a22 - a12 * a21;
    const T c01 = -(a10 * a22 - a12 * a20);
    const T c02 = a10 * a21 - a11 * a20;

    const T c10 = -(a01 * a22 - a02 * a21);
    const T c11 = a00 * a22 - a02 * a20;
    const T c12 = -(a00 * a21 - a01 * a20);

    const T c20 = a01 * a12 - a02 * a11;
    const T c21 = -(a00 * a12 - a02 * a10);
    const T c22 = a00 * a11 - a01 * a10;

    T det = a00 * c00 + a01 * c01 + a02 * c02;
    const T eps = T(1e-20f);
    if (det == T(0)) det = (det >= T(0)) ? eps : -eps;
    const T invdet = T(1) / det;

    i00 = c00 * invdet;  i01 = c10 * invdet;  i02 = c20 * invdet;
    i10 = c01 * invdet;  i11 = c11 * invdet;  i12 = c21 * invdet;
    i20 = c02 * invdet;  i21 = c12 * invdet;  i22 = c22 * invdet;
}

template <int MAX_C, typename T>
__global__ void cond_mean_convariance_opacity_fwd_kernel(
    const int64_t N,
    const int64_t D,
    const T* __restrict__ means,     // [N,D]
    const T* __restrict__ covars,    // [N,D,D]
    const T* __restrict__ opacities, // [N,1]
    const T* __restrict__ betas,     // [N,C]
    const T* __restrict__ query,     // [N,C]
    T* __restrict__ out_means,       // [N,3]
    T* __restrict__ out_covars,      // [N,3,3]
    T* __restrict__ out_opacities    // [N,1]
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int C = int(D - 3);
    if (C <= 0 || C > MAX_C) return;

    const T* mptr = means  + idx * D;
    const T* vptr = covars + idx * D * D;
    const T  o_in = opacities[idx];
    const T* bptr = betas  + idx * C;
    const T* qptr = query  + idx * C;

    T* om = out_means     + idx * 3;
    T* oV = out_covars    + idx * 9;
    T* oo = out_opacities + idx;

    // m1 and m2
    const T m1x = mptr[0], m1y = mptr[1], m1z = mptr[2];
    T m2[MAX_C];
#pragma unroll
    for (int j = 0; j < C; ++j) m2[j] = mptr[3 + j];

    // x = q - m2, betas and beta_adj
    T x[MAX_C], beta[MAX_C], beta_adj[MAX_C];
#pragma unroll
    for (int j = 0; j < C; ++j) {
        x[j] = qptr[j] - m2[j];
        beta[j] = bptr[j];
        T t = beta[j] * T(0.25f);
        beta_adj[j] = (t < T(1)) ? t : T(1);
    }

    // v11 [3x3]
    T v11[9];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int c = 0; c < 3; ++c)
            v11[r * 3 + c] = vptr[r * D + c];

    // v12 [3xC], v21 [Cx3], v22 [CxC]
    T v12[3 * MAX_C], v21[MAX_C * 3], v22[MAX_C * MAX_C];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c)
            v12[r * C + c] = vptr[r * D + (3 + c)];
#pragma unroll
    for (int r = 0; r < C; ++r)
#pragma unroll
        for (int c = 0; c < 3; ++c)
            v21[r * 3 + c] = vptr[(3 + r) * D + c];
#pragma unroll
    for (int r = 0; r < C; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c)
            v22[r * C + c] = vptr[(3 + r) * D + (3 + c)];

    // v22_inv (fast 3x3 path; generic for C!=3 up to MAX_C)
    T i22[MAX_C * MAX_C];
    if (C == 3) {
        T i00,i01,i02,i10,i11,i12,i20,i21,i22_;
        invert_3x3(
            v22[0], v22[1], v22[2],
            v22[3], v22[4], v22[5],
            v22[6], v22[7], v22[8],
            i00,i01,i02,i10,i11,i12,i20,i21,i22_);
        i22[0]=i00; i22[1]=i01; i22[2]=i02;
        i22[3]=i10; i22[4]=i11; i22[5]=i12;
        i22[6]=i20; i22[7]=i21; i22[8]=i22_;
    } else {
        invert_small_matrix<MAX_C, T>(v22, i22, C);
    }

    // v_regr = v12 @ i22   [3xC]
    T r_[3 * MAX_C];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            T acc = T(0);
#pragma unroll
            for (int k = 0; k < C; ++k) acc += v12[r * C + k] * i22[k * C + c];
            r_[r * C + c] = acc;
        }
    }

    // r_beta = r_ * beta_adj (column-wise)
    T rb[3 * MAX_C];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c)
            rb[r * C + c] = r_[r * C + c] * beta_adj[c];

    // m_change = rb @ x
    T mcx=T(0), mcy=T(0), mcz=T(0);
#pragma unroll
    for (int c = 0; c < C; ++c) {
        mcx += rb[0 * C + c] * x[c];
        mcy += rb[1 * C + c] * x[c];
        mcz += rb[2 * C + c] * x[c];
    }
    om[0] = m1x + mcx; om[1] = m1y + mcy; om[2] = m1z + mcz;

    // v_change = rb @ v21
    T c00=T(0),c01=T(0),c02=T(0),
      c10=T(0),c11=T(0),c12=T(0),
      c20=T(0),c21=T(0),c22=T(0);
#pragma unroll
    for (int k = 0; k < C; ++k) {
        const T rb0 = rb[0 * C + k], rb1 = rb[1 * C + k], rb2 = rb[2 * C + k];
        const T v210 = v21[k * 3 + 0], v211 = v21[k * 3 + 1], v212 = v21[k * 3 + 2];
        c00 += rb0 * v210; c01 += rb0 * v211; c02 += rb0 * v212;
        c10 += rb1 * v210; c11 += rb1 * v211; c12 += rb1 * v212;
        c20 += rb2 * v210; c21 += rb2 * v211; c22 += rb2 * v212;
    }
    // v_cond = v11 - v_change
    oV[0]=v11[0]-c00; oV[1]=v11[1]-c01; oV[2]=v11[2]-c02;
    oV[3]=v11[3]-c10; oV[4]=v11[4]-c11; oV[5]=v11[5]-c12;
    oV[6]=v11[6]-c20; oV[7]=v11[7]-c21; oV[8]=v11[8]-c22;

    T Lc[MAX_C * MAX_C];
    #pragma unroll
        for (int r = 0; r < MAX_C; ++r) {
    #pragma unroll
            for (int c = 0; c < MAX_C; ++c) {
                Lc[r * MAX_C + c] = T(0);
            }
        }

    // Compute Lc for n=C: v22 = Lc * Lc^T
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j <= i; ++j) {
            T sum = v22[i * C + j];
            for (int k = 0; k < j; ++k) {
                sum -= Lc[i * MAX_C + k] * Lc[j * MAX_C + k];
            }
            if (i == j) {
                float s = float(sum);
                // numeric guard: SPD expected, but protect anyway
                if (s <= 0.f) s = 1e-20f;
                Lc[i * MAX_C + j] = T(sqrtf(s));
            } else {
                T denom = Lc[j * MAX_C + j];
                if (denom == T(0)) denom = T(1e-20f);
                Lc[i * MAX_C + j] = sum / denom;
            }
        }
    }

    // Forward solve Lc * y = x
    T ychol[MAX_C];
#pragma unroll
    for (int i = 0; i < C; ++i) {
        T sum = x[i];
        for (int k = 0; k < i; ++k) sum -= Lc[i * MAX_C + k] * ychol[k];
        T denom = Lc[i * MAX_C + i];
        if (denom == T(0)) denom = T(1e-20f);
        ychol[i] = sum / denom;
    }

    T o_change = T(1);
#pragma unroll
    for (int i = 0; i < C; ++i) {
        T d = ychol[i] * ychol[i];
        d = ftanh(d);
        d = clamp01mEps(d);
        o_change *= fpow(T(1) - d, beta[i]);
    }
    oo[0] = o_in * o_change;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cond_mean_convariance_opacity_fwd_tensor(
    const torch::Tensor& means,     // [N,D]
    const torch::Tensor& covars,    // [N,D,D]
    const torch::Tensor& opacities, // [N,1]
    const torch::Tensor& betas,     // [N,C]
    const torch::Tensor& query      // [N,C]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(betas);
    GSPLAT_CHECK_INPUT(query);

    TORCH_CHECK(means.dim() == 2, "means must be [N,D]");
    TORCH_CHECK(covars.dim() == 3, "covars must be [N,D,D]");
    TORCH_CHECK(opacities.dim() == 2 && opacities.size(1) == 1, "opacities must be [N,1]");
    TORCH_CHECK(betas.dim() == 2 && query.dim() == 2, "betas/query must be [N,C]");

    const auto N = means.size(0);
    const auto D = means.size(1);
    TORCH_CHECK(covars.size(0) == N && covars.size(1) == D && covars.size(2) == D, "covars must match [N,D,D]");
    TORCH_CHECK(query.size(0) == N && betas.size(0) == N, "batch sizes must match");
    TORCH_CHECK(D >= 4, "D must be >= 4");
    TORCH_CHECK(int64_t(D - 3) == betas.size(1) && betas.size(1) == query.size(1), "C must be D-3");

    auto opts = means.options();
    auto out_means     = torch::empty({N, 3}, opts);
    auto out_covars    = torch::empty({N, 3, 3}, opts);
    auto out_opacities = torch::empty({N, 1}, opts);

    if (N > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int kThreads = GSPLAT_N_THREADS;
        const dim3 grid((N + kThreads - 1) / kThreads);
        const dim3 block(kThreads);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            means.scalar_type(),
            "cond_mean_convariance_opacity_fwd", [&]() {
                cond_mean_convariance_opacity_fwd_kernel<8, scalar_t>  // MAX_C = 8
                    <<<grid, block, 0, stream>>>(
                        N, D,
                        means.data_ptr<scalar_t>(),
                        covars.data_ptr<scalar_t>(),
                        opacities.data_ptr<scalar_t>(),
                        betas.data_ptr<scalar_t>(),
                        query.data_ptr<scalar_t>(),
                        out_means.data_ptr<scalar_t>(),
                        out_covars.data_ptr<scalar_t>(),
                        out_opacities.data_ptr<scalar_t>());
            }
        );
    }
    return {out_means, out_covars, out_opacities};
}

} // namespace gsplat
