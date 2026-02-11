// cond_mean_convariance_opacity_bwd.cu  (MAX_C = 8, C==3 fast path)
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

template <typename T> __device__ inline T fexp(T x)  { return T(expf(float(x))); }
template <typename T> __device__ inline T flog(T x)  { return T(logf(float(x))); }
template <typename T> __device__ inline T fpow(T a, T b) { return T(powf(float(a), float(b))); }
template <typename T> __device__ inline T fsqrt(T x) { return T(sqrtf(float(x))); }
template <typename T> __device__ inline T ftanh(T x) { return T(tanhf(float(x))); }

template <int MAX_N, typename T>
__device__ inline void invert_small_matrix(const T* __restrict__ A, T* __restrict__ invA, int n) {
    T aug[(MAX_N) * (2*MAX_N)];
#pragma unroll
    for (int r = 0; r < MAX_N; ++r)
#pragma unroll
        for (int c = 0; c < 2*MAX_N; ++c) aug[r*(2*MAX_N) + c] = T(0);
#pragma unroll
    for (int r = 0; r < n; ++r) {
#pragma unroll
        for (int c = 0; c < n; ++c) aug[r*(2*MAX_N) + c] = A[r*n + c];
        aug[r*(2*MAX_N) + (n + r)] = T(1);
    }
    for (int col = 0; col < n; ++col) {
        int piv = col;
        T maxabs = fabsf(float(aug[piv*(2*MAX_N) + col]));
        for (int r = col+1; r < n; ++r) {
            T v = fabsf(float(aug[r*(2*MAX_N) + col]));
            if (v > maxabs) { maxabs = v; piv = r; }
        }
        if (piv != col) {
#pragma unroll
            for (int c = 0; c < 2*n; ++c) {
                T tmp = aug[col*(2*MAX_N) + c];
                aug[col*(2*MAX_N) + c] = aug[piv*(2*MAX_N) + c];
                aug[piv*(2*MAX_N) + c] = tmp;
            }
        }
        T diag = aug[col*(2*MAX_N) + col];
        if (diag == T(0)) { const T eps = T(1e-20f); diag = (diag >= T(0)) ? eps : -eps; }
        T invdiag = T(1) / diag;
#pragma unroll
        for (int c = 0; c < 2*n; ++c) aug[col*(2*MAX_N) + c] *= invdiag;
        for (int r = 0; r < n; ++r) if (r != col) {
            T f = aug[r*(2*MAX_N) + col];
            if (f != T(0)) {
#pragma unroll
                for (int c = 0; c < 2*n; ++c) aug[r*(2*MAX_N) + c] -= f * aug[col*(2*MAX_N) + c];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < n; ++r)
#pragma unroll
        for (int c = 0; c < n; ++c) invA[r*n + c] = aug[r*(2*MAX_N) + (n + c)];
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
    const T c02 =  a10 * a21 - a11 * a20;
    const T c10 = -(a01 * a22 - a02 * a21);
    const T c11 =  a00 * a22 - a02 * a20;
    const T c12 = -(a00 * a21 - a01 * a20);
    const T c20 =  a01 * a12 - a02 * a11;
    const T c21 = -(a00 * a12 - a02 * a10);
    const T c22 =  a00 * a11 - a01 * a10;

    T det = a00*c00 + a01*c01 + a02*c02;
    const T eps = T(1e-20f);
    if (det == T(0)) det = (det >= T(0)) ? eps : -eps;
    const T invdet = T(1) / det;

    i00 = c00 * invdet;  i01 = c10 * invdet;  i02 = c20 * invdet;
    i10 = c01 * invdet;  i11 = c11 * invdet;  i12 = c21 * invdet;
    i20 = c02 * invdet;  i21 = c12 * invdet;  i22 = c22 * invdet;
}

template <int MAX_C, typename T>
__global__ void cond_mean_convariance_opacity_bwd_kernel(
    const int64_t N,
    const int64_t D,                   // total dims
    // saved forward inputs
    const T* __restrict__ means,       // [N,D]
    const T* __restrict__ covars,      // [N,D,D]
    const T* __restrict__ opacities,   // [N,1]
    const T* __restrict__ betas,       // [N,C]
    const T* __restrict__ query,       // [N,C]
    // upstream grads
    const T* __restrict__ v_mcond,     // [N,3]
    const T* __restrict__ v_vcond,     // [N,3,3]
    const T* __restrict__ v_ocond,     // [N,1]
    // outputs
    T* __restrict__ g_means,           // [N,D]
    T* __restrict__ g_covars,          // [N,D,D]
    T* __restrict__ g_opacities,       // [N,1]
    T* __restrict__ g_betas            // [N,C]
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int C = int(D - 3);
    if (C <= 0 || C > MAX_C) return;

    // Base pointers
    const T* mptr = means  + idx * D;
    const T* vptr = covars + idx * D * D;
    const T  opa  = opacities[idx];
    const T* bptr = betas  + idx * C;
    const T* qptr = query  + idx * C;

    const T* gM = v_mcond + idx * 3;
    const T* gV = v_vcond + idx * 9;
    const T  gO = v_ocond[idx];

    T* gm = g_means  + idx * D;
    T* gVfull = g_covars + idx * D * D;
    T* go = g_opacities + idx;
    T* gb = g_betas + idx * C;

#pragma unroll
    for (int i=0;i<D;++i) gm[i] = T(0);
#pragma unroll
    for (int i=0;i<D*D;++i) gVfull[i] = T(0);
    *go = T(0);
#pragma unroll
    for (int i=0;i<C;++i) gb[i] = T(0);

    // m1, m2, x, betas
    const T m1x = mptr[0], m1y = mptr[1], m1z = mptr[2];
    (void)m1x; (void)m1y; (void)m1z;
    T m2[MAX_C], beta[MAX_C], beta_adj[MAX_C], x[MAX_C];
#pragma unroll
    for (int j=0;j<C;++j) {
        m2[j] = mptr[3 + j];
        beta[j] = bptr[j];
        T t = beta[j] * T(0.25f);
        beta_adj[j] = (t < T(1)) ? t : T(1);
        x[j] = qptr[j] - m2[j];
    }

    // V blocks
    T v11[9];
#pragma unroll
    for (int r=0;r<3;++r)
#pragma unroll
        for (int c=0;c<3;++c)
            v11[r*3 + c] = vptr[r*D + c];
    (void)v11;

    T v12[3*MAX_C], v21[MAX_C*3], v22[MAX_C*MAX_C];
#pragma unroll
    for (int r=0;r<3;++r)
#pragma unroll
        for (int c=0;c<C;++c)
            v12[r*C + c] = vptr[r*D + (3 + c)];
#pragma unroll
    for (int r=0;r<C;++r)
#pragma unroll
        for (int c=0;c<3;++c)
            v21[r*3 + c] = vptr[(3 + r)*D + c];
#pragma unroll
    for (int r=0;r<C;++r)
#pragma unroll
        for (int c=0;c<C;++c)
            v22[r*C + c] = vptr[(3 + r)*D + (3 + c)];

    // i22
    T i22[MAX_C*MAX_C];
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

    // r = v12 @ i22
    T r_[3*MAX_C];
#pragma unroll
    for (int rr=0; rr<3; ++rr) {
#pragma unroll
        for (int cc=0; cc<C; ++cc) {
            T acc = T(0);
#pragma unroll
            for (int k=0; k<C; ++k) acc += v12[rr*C + k] * i22[k*C + cc];
            r_[rr*C + cc] = acc;
        }
    }

    // rb = r * beta_adj
    T rb[3*MAX_C];
#pragma unroll
    for (int rr=0; rr<3; ++rr)
#pragma unroll
        for (int cc=0; cc<C; ++cc)
            rb[rr*C + cc] = r_[rr*C + cc] * beta_adj[cc];

    // ===== upstream
    const T gmx = gM[0], gmy = gM[1], gmz = gM[2];
    const T Gv00 = gV[0], Gv01 = gV[1], Gv02 = gV[2];
    const T Gv10 = gV[3], Gv11 = gV[4], Gv12 = gV[5];
    const T Gv20 = gV[6], Gv21 = gV[7], Gv22 = gV[8];

    // ===== m_cond path
    gm[0] += gmx; gm[1] += gmy; gm[2] += gmz;

    // Gr := ∂L/∂r_beta  (init with mean path: gM ⊗ x^T)
    T Gr[3*MAX_C];
#pragma unroll
    for (int cc=0; cc<C; ++cc) {
        Gr[0*C + cc] = gmx * x[cc];
        Gr[1*C + cc] = gmy * x[cc];
        Gr[2*C + cc] = gmz * x[cc];
    }

    // gx from m_cond: rb^T @ gM
    T gx[MAX_C]; for (int i=0;i<C;++i) gx[i] = T(0);
#pragma unroll
    for (int cc=0; cc<C; ++cc) {
        gx[cc] += rb[0*C + cc] * gmx
                + rb[1*C + cc] * gmy
                + rb[2*C + cc] * gmz;
    }

    // ===== v_cond path
    // v11: +Gv
    gVfull[0*D + 0] += Gv00; gVfull[0*D + 1] += Gv01; gVfull[0*D + 2] += Gv02;
    gVfull[1*D + 0] += Gv10; gVfull[1*D + 1] += Gv11; gVfull[1*D + 2] += Gv12;
    gVfull[2*D + 0] += Gv20; gVfull[2*D + 1] += Gv21; gVfull[2*D + 2] += Gv22;

    // Gr add from v_cond: - Gv @ v21^T
#pragma unroll
    for (int cc=0; cc<C; ++cc) {
        const T a0 = v21[cc*3 + 0], a1 = v21[cc*3 + 1], a2 = v21[cc*3 + 2];
        Gr[0*C + cc] += -(Gv00 * a0 + Gv01 * a1 + Gv02 * a2);
        Gr[1*C + cc] += -(Gv10 * a0 + Gv11 * a1 + Gv12 * a2);
        Gr[2*C + cc] += -(Gv20 * a0 + Gv21 * a1 + Gv22 * a2);
    }

    // v21 grad: - rb^T @ Gv
#pragma unroll
    for (int rr=0; rr<C; ++rr) {
        const T b0 = rb[0*C + rr], b1 = rb[1*C + rr], b2 = rb[2*C + rr];
        gVfull[(3+rr)*D + 0] += -(b0 * Gv00 + b1 * Gv10 + b2 * Gv20);
        gVfull[(3+rr)*D + 1] += -(b0 * Gv01 + b1 * Gv11 + b2 * Gv21);
        gVfull[(3+rr)*D + 2] += -(b0 * Gv02 + b1 * Gv12 + b2 * Gv22);
    }

    // ===== Chain r_beta -> r and beta_adj
    // dL/dbeta_adj[i] = Σ_j Gr[j,i] * r[j,i]
    T dL_dba[MAX_C]; for (int i=0;i<C;++i) dL_dba[i] = T(0);
#pragma unroll
    for (int cc=0; cc<C; ++cc) {
        dL_dba[cc] = Gr[0*C + cc] * r_[0*C + cc]
                   + Gr[1*C + cc] * r_[1*C + cc]
                   + Gr[2*C + cc] * r_[2*C + cc];
    }

    // dL/dr = Gr * diag(beta_adj)
    T G_r[3*MAX_C];
#pragma unroll
    for (int rr=0; rr<3; ++rr)
#pragma unroll
        for (int cc=0; cc<C; ++cc)
            G_r[rr*C + cc] = Gr[rr*C + cc] * beta_adj[cc];

    // v12 grad: G_r @ i22^T
#pragma unroll
    for (int rr=0; rr<3; ++rr)
#pragma unroll
        for (int cc=0; cc<C; ++cc) {
            T acc = T(0);
#pragma unroll
            for (int k=0; k<C; ++k) acc += G_r[rr*C + k] * i22[cc*C + k];
            gVfull[rr*D + (3 + cc)] += acc;
        }

    // Gi22 from r-path: v12^T @ G_r
    T Gi22[MAX_C*MAX_C]; for (int i=0;i<C*C;++i) Gi22[i] = T(0);
#pragma unroll
    for (int rr=0; rr<C; ++rr)
#pragma unroll
        for (int cc=0; cc<C; ++cc) {
            T acc = T(0);
#pragma unroll
            for (int k=0; k<3; ++k) acc += v12[k*C + rr] * G_r[k*C + cc];
            Gi22[rr*C + cc] += acc;
        }

    // ===================== Opacity path =====================
    //   PyTorch:
    //   L = cholesky(v22) lower
    //   y = solve_triangular(L, x)
    //   d = tanh(y^2), clamp
    //   o_change = Π (1-d)^beta
    //

    // ---- (1) Cholesky factor Lc (lower) of v22
    T Lc[MAX_C*MAX_C];
#pragma unroll
    for (int r=0; r<MAX_C; ++r)
#pragma unroll
        for (int c=0; c<MAX_C; ++c)
            Lc[r*MAX_C + c] = T(0);

    for (int i=0; i<C; ++i) {
        for (int j=0; j<=i; ++j) {
            T sum = v22[i*C + j];
            for (int k=0; k<j; ++k) sum -= Lc[i*MAX_C + k] * Lc[j*MAX_C + k];
            if (i == j) {
                float s = float(sum);
                if (s <= 0.f) s = 1e-20f;          // guard
                Lc[i*MAX_C + j] = T(sqrtf(s));
            } else {
                T denom = Lc[j*MAX_C + j];
                if (denom == T(0)) denom = T(1e-20f);
                Lc[i*MAX_C + j] = sum / denom;
            }
        }
    }

    // ---- (2) Forward solve: Lc * y = x
    T ychol[MAX_C];
#pragma unroll
    for (int i=0; i<C; ++i) {
        T sum = x[i];
        for (int k=0; k<i; ++k) sum -= Lc[i*MAX_C + k] * ychol[k];
        T denom = Lc[i*MAX_C + i];
        if (denom == T(0)) denom = T(1e-20f);
        ychol[i] = sum / denom;
    }

    // ---- (3) Forward compute d, base, o_change
    T d_i[MAX_C], base[MAX_C], o_change = T(1);
    const T upper = T(1) - T(std::numeric_limits<T>::epsilon());
#pragma unroll
    for (int i=0; i<C; ++i) {
        T d = ftanh(ychol[i] * ychol[i]);  // tanh(y^2) in [0,1)
        // clamp to [0, 1-eps]
        if (d < T(0)) d = T(0);
        if (d > upper) d = upper;
        d_i[i] = d;
        base[i] = T(1) - d;
        o_change *= fpow(base[i], beta[i]);
    }

    // o_cond = opa * o_change
    *go += gO * o_change;
    const T g_o_change = gO * opa;

    // ---- (4) Per-dim grads wrt beta and d
    T g_d[MAX_C]; for (int i=0;i<C;++i) g_d[i] = T(0);
#pragma unroll
    for (int i=0; i<C; ++i) {
        if (base[i] > T(0)) {
            // ∂/∂beta_i: o_change * log(base_i)
            gb[i] += g_o_change * o_change * flog(base[i]);

            // ∂/∂d_i: o_change * beta_i * (-1/base_i), respecting clamp
            const bool active = (d_i[i] > T(0)) && (d_i[i] < upper);
            if (active) {
                g_d[i] += g_o_change * (-o_change * beta[i] / base[i]);
            }
        }
    }

    // ---- (5) d = tanh(y^2) => ∂d/∂y = 2*y*(1 - d^2)
    T g_y[MAX_C]; for (int i=0;i<C;++i) g_y[i] = T(0);
#pragma unroll
    for (int i=0; i<C; ++i) {
        // if clamped, g_d already 0
        const T d = d_i[i];
        const T dd_dy = T(2) * ychol[i] * (T(1) - d * d);
        g_y[i] = g_d[i] * dd_dy;
    }

    // ---- (6) y = L^{-1} x
    // g_x += L^{-T} g_y
    // g_L = - (L^{-T} g_y) ⊗ y^T
    //
    // Compute a = L^{-T} g_y by solving (L^T) a = g_y
    T a[MAX_C];
    for (int i=C-1; i>=0; --i) {
        T sum = g_y[i];
        for (int k=i+1; k<C; ++k) sum -= Lc[k*MAX_C + i] * a[k]; // L^T(i,k) = L(k,i)
        T denom = Lc[i*MAX_C + i];
        if (denom == T(0)) denom = T(1e-20f);
        a[i] = sum / denom;
    }

    // Add to gx (shared accumulator for x from mean/cov + opacity)
#pragma unroll
    for (int i=0; i<C; ++i) gx[i] += a[i];

    // gL = - a ⊗ y^T  (lower-tri entries only)
    T gL[MAX_C*MAX_C];
#pragma unroll
    for (int r=0; r<MAX_C; ++r)
#pragma unroll
        for (int c=0; c<MAX_C; ++c)
            gL[r*MAX_C + c] = T(0);

#pragma unroll
    for (int r=0; r<C; ++r)
#pragma unroll
        for (int c=0; c<=r; ++c)
            gL[r*MAX_C + c] = -(a[r] * ychol[c]);

    // ---- (7) Backprop through Cholesky: v22 = L L^T
    // Use standard chol backward:
    //   S = L^T gL
    //   S = tril(S); diag *= 0.5
    //   G = L^{-T} S L^{-1}
    //   gA = 0.5*(G + G^T)
    //
    // NOTE: L here is lower; we stored it in Lc with stride MAX_C.
    T S[MAX_C*MAX_C];
#pragma unroll
    for (int r=0; r<MAX_C; ++r)
#pragma unroll
        for (int c=0; c<MAX_C; ++c)
            S[r*MAX_C + c] = T(0);

    // S = L^T gL
#pragma unroll
    for (int r=0; r<C; ++r) {
#pragma unroll
        for (int c=0; c<C; ++c) {
            T acc = T(0);
#pragma unroll
            for (int k=0; k<C; ++k) {
                // L^T(r,k) = L(k,r)
                acc += Lc[k*MAX_C + r] * gL[k*MAX_C + c];
            }
            S[r*MAX_C + c] = acc;
        }
    }

    // S = tril(S); diag *= 0.5
#pragma unroll
    for (int r=0; r<C; ++r) {
#pragma unroll
        for (int c=0; c<C; ++c) {
            if (c > r) S[r*MAX_C + c] = T(0);
        }
        S[r*MAX_C + r] *= T(0.5f);
    }

    // U = L^{-T} S  => solve (L^T) U = S for U (columns)
    // We'll solve column-by-column: (L^T) u_col = s_col (upper-tri solve)
    T Umat[MAX_C*MAX_C];
#pragma unroll
    for (int r=0; r<MAX_C; ++r)
#pragma unroll
        for (int c=0; c<MAX_C; ++c)
            Umat[r*MAX_C + c] = T(0);

#pragma unroll
    for (int col=0; col<C; ++col) {
        // back substitution since L^T is upper
        for (int i=C-1; i>=0; --i) {
            T sum = S[i*MAX_C + col];
            for (int k=i+1; k<C; ++k) sum -= Lc[k*MAX_C + i] * Umat[k*MAX_C + col];
            T denom = Lc[i*MAX_C + i];
            if (denom == T(0)) denom = T(1e-20f);
            Umat[i*MAX_C + col] = sum / denom;
        }
    }

    // G = U * L^{-1}  => for each row of U, right-multiply by L^{-1}
    // Implement as: solve L * W^T = U^T  (forward solve), then G = W^T
    T Gmat[MAX_C*MAX_C];
#pragma unroll
    for (int r=0; r<MAX_C; ++r)
#pragma unroll
        for (int c=0; c<MAX_C; ++c)
            Gmat[r*MAX_C + c] = T(0);

#pragma unroll
    for (int col=0; col<C; ++col) { // solving for W(:,col) where W^T(:,col) corresponds to row col of G
        // Solve L * w = U^T(:,col) => w = L^{-1} U^T(:,col)
        for (int i=0; i<C; ++i) {
            T sum = Umat[col*MAX_C + i]; // U^T(i,col) = U(col,i)
            for (int k=0; k<i; ++k) sum -= Lc[i*MAX_C + k] * Gmat[k*MAX_C + col];
            T denom = Lc[i*MAX_C + i];
            if (denom == T(0)) denom = T(1e-20f);
            Gmat[i*MAX_C + col] = sum / denom;
        }
    }

    // symmetrize: gA = 0.5*(G + G^T)
#pragma unroll
    for (int r=0; r<C; ++r)
#pragma unroll
        for (int c=0; c<C; ++c) {
            const T gA = T(0.5f) * (Gmat[r*MAX_C + c] + Gmat[c*MAX_C + r]);
            gVfull[(3+r)*D + (3+c)] += gA;
        }

    // ===================== End Opacity path =====================

    // ===== Means from x (x = q - m2)
#pragma unroll
    for (int j=0;j<C;++j) gm[3 + j] += -gx[j];

    // ===== betas: chain beta_adj (mean/cov part)
#pragma unroll
    for (int i=0;i<C;++i) {
        const T dba_dbeta = (beta[i] < T(4)) ? T(0.25f) : T(0);
        gb[i] += dL_dba[i] * dba_dbeta;
    }

    // ===== Map Gi22 -> Gv22 for mean/cov conditioning ONLY (unchanged)
    // Gv22_from_inv = - i22^T * Gi22 * i22^T
    T tmp[MAX_C*MAX_C]; for (int i=0;i<C*C;++i) tmp[i] = T(0);
#pragma unroll
    for (int r=0; r<C; ++r)
#pragma unroll
        for (int c=0; c<C; ++c) {
            T acc = T(0);
#pragma unroll
            for (int k=0; k<C; ++k) acc += Gi22[r*C + k] * i22[c*C + k]; // Gi22 * i22^T
            tmp[r*C + c] = acc;
        }
#pragma unroll
    for (int r=0; r<C; ++r)
#pragma unroll
        for (int c=0; c<C; ++c) {
            T acc = T(0);
#pragma unroll
            for (int k=0; k<C; ++k) acc += i22[k*C + r] * tmp[k*C + c];   // i22^T * tmp
            gVfull[(3+r)*D + (3+c)] += -acc;
        }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
cond_mean_convariance_opacity_bwd_tensor(
    const torch::Tensor& means,     // [N,D]
    const torch::Tensor& covars,    // [N,D,D]
    const torch::Tensor& opacities, // [N,1]
    const torch::Tensor& betas,     // [N,C]
    const torch::Tensor& query,     // [N,C]
    const torch::Tensor& v_mcond,   // [N,3]
    const torch::Tensor& v_vcond,   // [N,3,3]
    const torch::Tensor& v_ocond    // [N,1]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(betas);
    GSPLAT_CHECK_INPUT(query);
    GSPLAT_CHECK_INPUT(v_mcond);
    GSPLAT_CHECK_INPUT(v_vcond);
    GSPLAT_CHECK_INPUT(v_ocond);

    const auto N = means.size(0);
    const auto D = means.size(1);
    TORCH_CHECK(covars.size(0)==N && covars.size(1)==D && covars.size(2)==D, "covars must match [N,D,D]");
    TORCH_CHECK(opacities.size(0)==N && opacities.size(1)==1, "opacities must be [N,1]");
    TORCH_CHECK(betas.size(0)==N && query.size(0)==N, "batch sizes must match");
    TORCH_CHECK(betas.dim()==2 && query.dim()==2, "betas/query must be [N,C]");
    TORCH_CHECK(int64_t(D - 3) == betas.size(1) && betas.size(1) == query.size(1), "C must be D-3");
    TORCH_CHECK(D >= 4, "D must be >= 4");

    auto opts = means.options();
    auto g_means     = torch::zeros({N, D},   opts);
    auto g_covars    = torch::zeros({N, D, D},opts);
    auto g_opacities = torch::zeros({N, 1},   opts);
    auto g_betas     = torch::zeros({N, betas.size(1)}, opts);

    if (N > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        constexpr int kThreads = GSPLAT_N_THREADS;
        const dim3 grid((N + kThreads - 1) / kThreads);
        const dim3 block(kThreads);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            means.scalar_type(),
            "cond_mean_convariance_opacity_bwd", [&]() {
                cond_mean_convariance_opacity_bwd_kernel<8, scalar_t>
                    <<<grid, block, 0, stream>>>(
                        N, D,
                        means.data_ptr<scalar_t>(),
                        covars.data_ptr<scalar_t>(),
                        opacities.data_ptr<scalar_t>(),
                        betas.data_ptr<scalar_t>(),
                        query.data_ptr<scalar_t>(),
                        v_mcond.data_ptr<scalar_t>(),
                        v_vcond.data_ptr<scalar_t>(),
                        v_ocond.data_ptr<scalar_t>(),
                        g_means.data_ptr<scalar_t>(),
                        g_covars.data_ptr<scalar_t>(),
                        g_opacities.data_ptr<scalar_t>(),
                        g_betas.data_ptr<scalar_t>());
            }
        );
    }
    return {g_means, g_covars, g_opacities, g_betas};
}

} // namespace gsplat
