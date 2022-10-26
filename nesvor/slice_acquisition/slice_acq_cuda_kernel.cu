#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

/*
template <typename scalar_t>
__device__ __forceinline__ scalar_t psf_func(const scalar_t x, const scalar_t y, const scalar_t z) {
    return expf(-0.5 * (x*x)/(1.2*1.2)) * expf(-0.5 * (y*y)/(1.2*1.2)) * expf(-0.5 * (z*z)/(3*3));
	//return expf(-0.5 *((x*x)/(1.2*1.2) + (y*y)/(1.2*1.2) + (z*z)/(3*3)));
}*/

template <typename scalar_t>
__global__ void slice_acquisition_forward_cuda_kernel( 
    const scalar_t* __restrict__ transforms,
    const scalar_t* __restrict__ vol,
    const bool* __restrict__ vol_mask,
    const scalar_t* __restrict__ psf,
    scalar_t* __restrict__ slices,
    const bool* __restrict__ slices_mask,
    scalar_t* __restrict__ slices_weight,
    const int32_t D, const int32_t H, const int32_t W,
    const int32_t d_p, const int32_t h_p, const int32_t w_p,
    const int32_t n, const int32_t h, const int32_t w,
    const scalar_t res_slice,
    const bool interp_psf
) {

    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n*h*w) return;
    if (slices_mask != NULL && !slices_mask[idx]) return;
    const int32_t Sy = W, Sz = H*W;

    const int32_t ix = idx % w;
    const int32_t iy = (idx / w) % h;
    const int32_t in = idx / (h * w); 

    const scalar_t r11 = transforms[in*12  ], r12 = transforms[in*12+1], r13 = transforms[in*12+2];
    const scalar_t r21 = transforms[in*12+4], r22 = transforms[in*12+5], r23 = transforms[in*12+6];
    const scalar_t r31 = transforms[in*12+8], r32 = transforms[in*12+9], r33 = transforms[in*12+10];

    scalar_t _x = (ix - (w - 1) / 2.) * res_slice + transforms[in*12+3];
    scalar_t _y = (iy - (h - 1) / 2.) * res_slice + transforms[in*12+7];
    scalar_t _z = transforms[in*12+11];

    scalar_t x_center = r11 * _x + r12 * _y + r13 * _z;
    scalar_t y_center = r21 * _x + r22 * _y + r23 * _z;
    scalar_t z_center = r31 * _x + r32 * _y + r33 * _z;

    x_center += (W - 1) / 2.;
    y_center += (H - 1) / 2.;
    z_center += (D - 1) / 2.;

    scalar_t val = 0;
    scalar_t weight = 0;
   
    for (int iz_p = -d_p/2, i_p = 0; iz_p < (d_p+1)/2; iz_p++) {
        for (int iy_p = -h_p/2; iy_p < (h_p+1)/2; iy_p++) {
            for (int ix_p = -w_p/2; ix_p < (w_p+1)/2; ix_p++, i_p++) {
                scalar_t psf_val = psf[i_p];
                if (psf_val == 0) continue;
                scalar_t x = x_center + r11*ix_p + r12*iy_p + r13*iz_p;
                scalar_t y = y_center + r21*ix_p + r22*iy_p + r23*iz_p;
                scalar_t z = z_center + r31*ix_p + r32*iy_p + r33*iz_p;
                if (x < 0 || y < 0 || z < 0 || x >= W-1 || y >= H-1 || z >= D-1) continue;
                scalar_t val_ = 0;
                if (interp_psf) { // NN
                    int32_t x_round = round(x);
                    int32_t y_round = round(y);
                    int32_t z_round = round(z);
                    int32_t i_v = z_round * Sz + y_round * Sy + x_round;
                    if (vol_mask != NULL && !vol_mask[i_v]) continue;
                    val_ = vol[i_v];

                    scalar_t x_psf = r11 * (x_round - x_center) + r21 * (y_round - y_center) + r31 * (z_round - z_center) + (w_p-1)/2.;
                    scalar_t y_psf = r12 * (x_round - x_center) + r22 * (y_round - y_center) + r32 * (z_round - z_center) + (h_p-1)/2.;
                    scalar_t z_psf = r13 * (x_round - x_center) + r23 * (y_round - y_center) + r33 * (z_round - z_center) + (d_p-1)/2.;

                    if (x_psf < 0 || y_psf < 0 || z_psf < 0 || x_psf >= w_p-1 || y_psf >= h_p-1 || z_psf >= d_p-1) continue;

                    int32_t x_floor = floor(x_psf);
                    int32_t y_floor = floor(y_psf);
                    int32_t z_floor = floor(z_psf);
                    scalar_t wx = x_psf - x_floor;
                    scalar_t wy = y_psf - y_floor;
                    scalar_t wz = z_psf - z_floor;
                    
                    i_v = z_floor * w_p * h_p + y_floor * w_p + x_floor;
                    psf_val = 0;
                    psf_val += (1-wx) * (1-wy) * (1-wz) * psf[i_v];
                    psf_val += wx * (1-wy) * (1-wz) * psf[i_v + 1];
                    psf_val += (1-wx) * wy * (1-wz) * psf[i_v + w_p];
                    psf_val += (1-wx) * (1-wy) * wz * psf[i_v + w_p*h_p];
                    psf_val += wx * wy * (1-wz) * psf[i_v + 1 + w_p];
                    psf_val += wx * (1-wy) * wz * psf[i_v + 1 + w_p*h_p];
                    psf_val += (1-wx) * wy * wz * psf[i_v + w_p + w_p*h_p];
                    psf_val += wx * wy * wz * psf[i_v + w_p + w_p*h_p + 1];
                    /*
                        scalar_t x_psf = r11 * (x_round - x_center) + r21 * (y_round - y_center) + r31 * (z_round - z_center);
                        scalar_t y_psf = r12 * (x_round - x_center) + r22 * (y_round - y_center) + r32 * (z_round - z_center);
                        scalar_t z_psf = r13 * (x_round - x_center) + r23 * (y_round - y_center) + r33 * (z_round - z_center);
                        psf_val = psf_func(x_psf, y_psf, z_psf);
                    */
                    val += psf_val * val_;
                    weight += psf_val;
                } else { // linear
                    
                    int32_t x_floor = floor(x);
                    int32_t y_floor = floor(y);
                    int32_t z_floor = floor(z);
                    scalar_t wx = x - x_floor;
                    scalar_t wy = y - y_floor;
                    scalar_t wz = z - z_floor;
                    int32_t i_v = z_floor * Sz + y_floor * Sy + x_floor;
                    
                    scalar_t psf_val_ = 0;
                    if (vol_mask == NULL || vol_mask[i_v]) {
                        psf_val_ = (1-wx) * (1-wy) * (1-wz) * psf_val;
                        val += psf_val_ * vol[i_v];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + 1]) {
                        psf_val_ = wx * (1-wy) * (1-wz) * psf_val;
                        val += psf_val_ * vol[i_v + 1];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + Sy]) {
                        psf_val_ = (1-wx) * wy * (1-wz) * psf_val;
                        val += psf_val_ * vol[i_v + Sy];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + Sz]) {
                        psf_val_ = (1-wx) * (1-wy) * wz * psf_val;
                        val += psf_val_ * vol[i_v + Sz];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + 1 + Sy]) {
                        psf_val_ =  wx * wy * (1-wz) * psf_val;
                        val += psf_val_ * vol[i_v + 1 + Sy];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + 1 + Sz]) {
                        psf_val_ =  wx * (1-wy) * wz * psf_val;
                        val += psf_val_ * vol[i_v + 1 + Sz];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + Sy + Sz]) {
                        psf_val_ =  (1-wx) * wy * wz * psf_val;
                        val += psf_val_ * vol[i_v + Sy + Sz];
                        weight += psf_val_;
                    }
                    if (vol_mask == NULL || vol_mask[i_v + Sy + Sz + 1]) {
                        psf_val_ =  wx * wy * wz * psf_val;
                        val += psf_val_ * vol[i_v + Sy + Sz + 1];
                        weight += psf_val_;
                    }
                }
                //val += psf_val * val_;
                //weight += psf_val;
            }
        }
    }
    if (weight > 0) {
        slices[idx] = val / weight;
        if (slices_weight != NULL) slices_weight[idx] = weight;
    }
}

template <typename scalar_t>
__global__ void slice_acquisition_backward_cuda_kernel( 
    const scalar_t* __restrict__ transforms,
    const scalar_t* __restrict__ vol,
    const bool* __restrict__ vol_mask,
    const scalar_t* __restrict__ psf,
    const scalar_t* __restrict__ grad_slices,
    const bool* __restrict__ slices_mask,
    scalar_t* __restrict__ grad_vol,
    scalar_t* __restrict__ grad_transforms,
    const int32_t D, const int32_t H, const int32_t W,
    const int32_t d_p, const int32_t h_p, const int32_t w_p,
    const int32_t n, const int32_t h, const int32_t w,
    const scalar_t res_slice,
    const bool interp_psf
) {
    
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n*h*w) return;
    if (slices_mask != NULL && !slices_mask[idx]) return;
    scalar_t gs = grad_slices[idx];
    if (gs == 0) return;

    const int32_t Sy = W, Sz = H*W;

    const int32_t ix = idx % w;
    const int32_t iy = (idx / w) % h;
    const int32_t in = idx / (h * w); 

    const scalar_t r11 = transforms[in*12  ], r12 = transforms[in*12+1], r13 = transforms[in*12+2];
    const scalar_t r21 = transforms[in*12+4], r22 = transforms[in*12+5], r23 = transforms[in*12+6];
    const scalar_t r31 = transforms[in*12+8], r32 = transforms[in*12+9], r33 = transforms[in*12+10];

    scalar_t _x = (ix - (w - 1) / 2.) * res_slice + transforms[in*12+3];
    scalar_t _y = (iy - (h - 1) / 2.) * res_slice + transforms[in*12+7];
    scalar_t _z = transforms[in*12+11];

    scalar_t x_center = r11 * _x + r12 * _y + r13 * _z;
    scalar_t y_center = r21 * _x + r22 * _y + r23 * _z;
    scalar_t z_center = r31 * _x + r32 * _y + r33 * _z;

    x_center += (W - 1) / 2.;
    y_center += (H - 1) / 2.;
    z_center += (D - 1) / 2.;

    scalar_t weight = 0;

    for (int iz_p = -d_p/2, i_p = 0; iz_p < (d_p+1)/2; iz_p++) {
        for (int iy_p = -h_p/2; iy_p < (h_p+1)/2; iy_p++) {
            for (int ix_p = -w_p/2; ix_p < (w_p+1)/2; ix_p++, i_p++) {
                scalar_t psf_val = psf[i_p];
                if (psf_val == 0) continue;
                scalar_t x = x_center + r11*ix_p + r12*iy_p + r13*iz_p;
                scalar_t y = y_center + r21*ix_p + r22*iy_p + r23*iz_p;
                scalar_t z = z_center + r31*ix_p + r32*iy_p + r33*iz_p;
                if (x < 0 || y < 0 || z < 0 || x >= W-1 || y >= H-1 || z >= D-1) continue;
                if (interp_psf) { // NN
                    scalar_t x_round_c = round(x) - x_center;
                    scalar_t y_round_c = round(y) - y_center;
                    scalar_t z_round_c = round(z) - z_center;

                    scalar_t x_psf = r11 * x_round_c + r21 * y_round_c + r31 * z_round_c + (w_p-1)/2.;
                    scalar_t y_psf = r12 * x_round_c + r22 * y_round_c + r32 * z_round_c + (h_p-1)/2.;
                    scalar_t z_psf = r13 * x_round_c + r23 * y_round_c + r33 * z_round_c + (d_p-1)/2.;

                    if (x_psf < 0 || y_psf < 0 || z_psf < 0 || x_psf >= w_p-1 || y_psf >= h_p-1 || z_psf >= d_p-1) continue;

                    int32_t x_floor = floor(x_psf);
                    int32_t y_floor = floor(y_psf);
                    int32_t z_floor = floor(z_psf);
                    scalar_t wx = x_psf - x_floor;
                    scalar_t wy = y_psf - y_floor;
                    scalar_t wz = z_psf - z_floor;
                    int32_t i_v = z_floor * w_p * h_p + y_floor * w_p + x_floor;
                    psf_val = 0;
                    psf_val += (1-wx) * (1-wy) * (1-wz) * psf[i_v];
                    psf_val += wx * (1-wy) * (1-wz) * psf[i_v + 1];
                    psf_val += (1-wx) * wy * (1-wz) * psf[i_v + w_p];
                    psf_val += (1-wx) * (1-wy) * wz * psf[i_v + w_p*h_p];
                    psf_val += wx * wy * (1-wz) * psf[i_v + 1 + w_p];
                    psf_val += wx * (1-wy) * wz * psf[i_v + 1 + w_p*h_p];
                    psf_val += (1-wx) * wy * wz * psf[i_v + w_p + w_p*h_p];
                    psf_val += wx * wy * wz * psf[i_v + w_p + w_p*h_p + 1];

                } //else { // linear}
                weight += psf_val;
            }
        }
    }

    if (weight == 0) return;
    gs /= weight;
    scalar_t g_r11 = 0, g_r12 = 0, g_r13 = 0, g_tx = 0;
    scalar_t g_r21 = 0, g_r22 = 0, g_r23 = 0, g_ty = 0;
    scalar_t g_r31 = 0, g_r32 = 0, g_r33 = 0, g_tz = 0;

    for (int iz_p = -d_p/2, i_p = 0; iz_p < (d_p+1)/2; iz_p++) {
        for (int iy_p = -h_p/2; iy_p < (h_p+1)/2; iy_p++) {
            for (int ix_p = -w_p/2; ix_p < (w_p+1)/2; ix_p++, i_p++) {
                scalar_t psf_val = psf[i_p];
                if (psf_val == 0) continue;
                scalar_t x = x_center + r11*ix_p + r12*iy_p + r13*iz_p;
                scalar_t y = y_center + r21*ix_p + r22*iy_p + r23*iz_p;
                scalar_t z = z_center + r31*ix_p + r32*iy_p + r33*iz_p;
                if (x < 0 || y < 0 || z < 0 || x >= W-1 || y >= H-1 || z >= D-1) continue;
                scalar_t val_ = 0;
                if (interp_psf) { // NN
                    int32_t x_round = round(x);
                    int32_t y_round = round(y);
                    int32_t z_round = round(z);

                    scalar_t x_psf = r11 * (x_round - x_center) + r21 * (y_round - y_center) + r31 * (z_round - z_center) + (w_p-1)/2.;
                    scalar_t y_psf = r12 * (x_round - x_center) + r22 * (y_round - y_center) + r32 * (z_round - z_center) + (h_p-1)/2.;
                    scalar_t z_psf = r13 * (x_round - x_center) + r23 * (y_round - y_center) + r33 * (z_round - z_center) + (d_p-1)/2.;

                    // x_psf = R^T * (x_round - R * _x - W) + w_p
                    // = R^T * (x_round - W) - (T + c) + w_p
                    if (x_psf < 0 || y_psf < 0 || z_psf < 0 || x_psf >= w_p-1 || y_psf >= h_p-1 || z_psf >= d_p-1) continue;

                    int32_t x_floor = floor(x_psf);
                    int32_t y_floor = floor(y_psf);
                    int32_t z_floor = floor(z_psf);
                    scalar_t wx = x_psf - x_floor;
                    scalar_t wy = y_psf - y_floor;
                    scalar_t wz = z_psf - z_floor;
                    int32_t i_v = z_floor * w_p * h_p + y_floor * w_p + x_floor;

                    if (vol_mask != NULL && !vol_mask[z_round * Sz + y_round * Sy + x_round]) continue;
                    
                    if (grad_vol != NULL) {
                        psf_val = 0;
                        psf_val += (1-wx) * (1-wy) * (1-wz) * psf[i_v];
                        psf_val += wx * (1-wy) * (1-wz) * psf[i_v + 1];
                        psf_val += (1-wx) * wy * (1-wz) * psf[i_v + w_p];
                        psf_val += (1-wx) * (1-wy) * wz * psf[i_v + w_p*h_p];
                        psf_val += wx * wy * (1-wz) * psf[i_v + 1 + w_p];
                        psf_val += wx * (1-wy) * wz * psf[i_v + 1 + w_p*h_p];
                        psf_val += (1-wx) * wy * wz * psf[i_v + w_p + w_p*h_p];
                        psf_val += wx * wy * wz * psf[i_v + w_p + w_p*h_p + 1];
                        atomicAdd(grad_vol + z_round * Sz + y_round * Sy + x_round, psf_val * gs);
                    }

                    if (grad_transforms != NULL) {
                        scalar_t dx = 0, dy = 0, dz = 0, tmp = 0;

                        tmp = psf[i_v];
                        dx -= (1-wy) * (1-wz) * tmp;
                        dy -= (1-wx) * (1-wz) * tmp;
                        dz -= (1-wx) * (1-wy) * tmp;

                        tmp = psf[i_v + 1];
                        dx += (1-wy) * (1-wz) * tmp;
                        dy -= wx * (1-wz) * tmp;
                        dz -= wx * (1-wy) * tmp;

                        tmp = psf[i_v + w_p];
                        dx -= wy * (1-wz) * tmp;
                        dy += (1-wx) * (1-wz) * tmp;
                        dz -= (1-wx) * wy * tmp;

                        tmp = psf[i_v +  w_p*h_p];
                        dx -= (1-wy) * wz * tmp;
                        dy -= (1-wx) * wz * tmp;
                        dz += (1-wx) * (1-wy) * tmp;

                        tmp = psf[i_v + 1 + w_p];
                        dx += wy * (1-wz) * tmp;
                        dy += wx * (1-wz) * tmp;
                        dz -= wx * wy * tmp;

                        tmp = psf[i_v + 1 + w_p*h_p];
                        dx += (1-wy) * wz * tmp;
                        dy -= wx * wz * tmp;
                        dz += wx * (1-wy) * tmp;

                        tmp = psf[i_v + w_p + w_p*h_p];
                        dx -= wy * wz * tmp;
                        dy += (1-wx) * wz * tmp;
                        dz += (1-wx) * wy * tmp;

                        tmp = psf[i_v + w_p + w_p*h_p + 1];
                        dx += wy * wz * tmp;
                        dy += wx * wz * tmp;
                        dz += wx * wy * tmp;

                        val_ = vol[z_round * Sz + y_round * Sy + x_round];
                        tmp = gs * val_;
                        dx *= tmp;
                        dy *= tmp;
                        dz *= tmp;

                        g_r11 += dx * (x_round - (W - 1) / 2.), g_r12 += dy * (x_round - (W - 1) / 2.), g_r13 += dz * (x_round - (W - 1) / 2.);
                        g_r21 += dx * (y_round - (H - 1) / 2.), g_r22 += dy * (y_round - (H - 1) / 2.), g_r23 += dz * (y_round - (H - 1) / 2.);
                        g_r31 += dx * (z_round - (D - 1) / 2.), g_r32 += dy * (z_round - (D - 1) / 2.), g_r33 += dz * (z_round - (D - 1) / 2.);
                        g_tx -= dx;
                        g_ty -= dy;
                        g_tz -= dz;
                    }

                } else { // linear
                    
                    int32_t x_floor = floor(x);
                    int32_t y_floor = floor(y);
                    int32_t z_floor = floor(z);
                    scalar_t wx = x - x_floor;
                    scalar_t wy = y - y_floor;
                    scalar_t wz = z - z_floor;
                    int32_t i_v = z_floor * Sz + y_floor * Sy + x_floor;

                    psf_val *= gs;
                    if (grad_vol != NULL){
                        if (vol_mask == NULL || vol_mask[i_v]) atomicAdd(grad_vol + i_v, (1-wx) * (1-wy) * (1-wz) * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+1]) atomicAdd(grad_vol + i_v + 1, wx * (1-wy) * (1-wz) * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+Sy]) atomicAdd(grad_vol + i_v + Sy, (1-wx) * wy * (1-wz) * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+Sz]) atomicAdd(grad_vol + i_v + Sz, (1-wx) * (1-wy) * wz * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+1+Sy]) atomicAdd(grad_vol + i_v + 1 + Sy, wx * wy * (1-wz) * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+1+Sz]) atomicAdd(grad_vol + i_v + 1 + Sz, wx * (1-wy) * wz * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz]) atomicAdd(grad_vol + i_v + Sy + Sz, (1-wx) * wy * wz * psf_val);
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz+1]) atomicAdd(grad_vol + i_v + Sy + Sz + 1, wx * wy * wz *  psf_val);
                    }
                    
                    if (grad_transforms != NULL){
                        scalar_t dx = 0, dy = 0, dz = 0;
                        if (vol_mask == NULL || vol_mask[i_v]) {
                            val_ = psf_val * vol[i_v];
                            dx -= (1-wy) * (1-wz) * val_;
                            dy -= (1-wx) * (1-wz) * val_;
                            dz -= (1-wx) * (1-wy) * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+1]) {
                            val_ = psf_val * vol[i_v + 1];
                            dx += (1-wy) * (1-wz) * val_;
                            dy -= wx * (1-wz) * val_;
                            dz -= wx * (1-wy) * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sy]) {
                            val_ = psf_val * vol[i_v + Sy];
                            dx -= wy * (1-wz) * val_;
                            dy += (1-wx) * (1-wz) * val_;
                            dz -= (1-wx) * wy * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sz]) {
                            val_ = psf_val * vol[i_v + Sz];
                            dx -= (1-wy) * wz * val_;
                            dy -= (1-wx) * wz * val_;
                            dz += (1-wx) * (1-wy) * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+1+Sy]) {
                            val_ = psf_val * vol[i_v + Sy + 1];
                            dx += wy * (1-wz) * val_;
                            dy += wx * (1-wz) * val_;
                            dz -= wx * wy * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+1+Sz]) {
                            val_ = psf_val * vol[i_v + Sz + 1];
                            dx += (1-wy) * wz * val_;
                            dy -= wx * wz * val_;
                            dz += wx * (1-wy) * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz]) {
                            val_ = psf_val * vol[i_v + Sz + Sy];
                            dx -= wy * wz * val_;
                            dy += (1-wx) * wz * val_;
                            dz += (1-wx) * wy * val_;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz+1]) {
                            val_ = psf_val * vol[i_v + Sz + Sy + 1];
                            dx += wy * wz * val_;
                            dy += wx * wz * val_;
                            dz += wx * wy * val_;
                        }
                        g_r11 += dx * (_x + ix_p), g_r12 += dx * (_y + iy_p), g_r13 += dx * (_z + iz_p);
                        g_r21 += dy * (_x + ix_p), g_r22 += dy * (_y + iy_p), g_r23 += dy * (_z + iz_p);
                        g_r31 += dz * (_x + ix_p), g_r32 += dz * (_y + iy_p), g_r33 += dz * (_z + iz_p);
                        g_tx += dx * r11 + dy * r21 + dz * r31;
                        g_ty += dx * r12 + dy * r22 + dz * r32;
                        g_tz += dx * r13 + dy * r23 + dz * r33;
                    }
                }
            }
        }
    }

    if (grad_transforms != NULL){
        atomicAdd(grad_transforms + in * 12, g_r11);
        atomicAdd(grad_transforms + in * 12 + 1, g_r12);
        atomicAdd(grad_transforms + in * 12 + 2, g_r13);
        atomicAdd(grad_transforms + in * 12 + 3, g_tx);
        atomicAdd(grad_transforms + in * 12 + 4, g_r21);
        atomicAdd(grad_transforms + in * 12 + 5, g_r22);
        atomicAdd(grad_transforms + in * 12 + 6, g_r23);
        atomicAdd(grad_transforms + in * 12 + 7, g_ty);
        atomicAdd(grad_transforms + in * 12 + 8, g_r31);
        atomicAdd(grad_transforms + in * 12 + 9, g_r32);
        atomicAdd(grad_transforms + in * 12 + 10, g_r33);
        atomicAdd(grad_transforms + in * 12 + 11, g_tz);
    }
}

template <typename scalar_t>
__global__ void slice_acquisition_adjoint_forward_cuda_kernel( 
    const scalar_t* __restrict__ transforms,
    scalar_t* __restrict__ vol,
    scalar_t* __restrict__ vol_weight,
    const bool* __restrict__ vol_mask,
    const scalar_t* __restrict__ psf,
    const scalar_t* __restrict__ slices,
    const bool* __restrict__ slices_mask,
    const int32_t D, const int32_t H, const int32_t W,
    const int32_t d_p, const int32_t h_p, const int32_t w_p,
    const int32_t n, const int32_t h, const int32_t w,
    const scalar_t res_slice,
    const bool interp_psf
) {
    
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n*h*w) return;
    if (slices_mask != NULL && !slices_mask[idx]) return;
    const scalar_t s = slices[idx];
    //if (s == 0) return;

    const int32_t Sy = W, Sz = H*W;
    const int32_t ix = idx % w;
    const int32_t iy = (idx / w) % h;
    const int32_t in = idx / (h * w); 

    const scalar_t r11 = transforms[in*12  ], r12 = transforms[in*12+1], r13 = transforms[in*12+2];
    const scalar_t r21 = transforms[in*12+4], r22 = transforms[in*12+5], r23 = transforms[in*12+6];
    const scalar_t r31 = transforms[in*12+8], r32 = transforms[in*12+9], r33 = transforms[in*12+10];

    scalar_t _x = (ix - (w - 1) / 2.) * res_slice + transforms[in*12+3];
    scalar_t _y = (iy - (h - 1) / 2.) * res_slice + transforms[in*12+7];
    scalar_t _z = transforms[in*12+11];

    scalar_t x_center = r11 * _x + r12 * _y + r13 * _z;
    scalar_t y_center = r21 * _x + r22 * _y + r23 * _z;
    scalar_t z_center = r31 * _x + r32 * _y + r33 * _z;

    x_center += (W - 1) / 2.;
    y_center += (H - 1) / 2.;
    z_center += (D - 1) / 2.;

    scalar_t weight = 0;

    for (int iz_p = -d_p/2, i_p = 0; iz_p < (d_p+1)/2; iz_p++) {
        for (int iy_p = -h_p/2; iy_p < (h_p+1)/2; iy_p++) {
            for (int ix_p = -w_p/2; ix_p < (w_p+1)/2; ix_p++, i_p++) {
                scalar_t psf_val = psf[i_p];
                if (psf_val == 0) continue;
                scalar_t x = x_center + r11*ix_p + r12*iy_p + r13*iz_p;
                scalar_t y = y_center + r21*ix_p + r22*iy_p + r23*iz_p;
                scalar_t z = z_center + r31*ix_p + r32*iy_p + r33*iz_p;
                if (x < 0 || y < 0 || z < 0 || x >= W-1 || y >= H-1 || z >= D-1) continue;
                if (interp_psf) { // NN
                    scalar_t x_round_c = round(x) - x_center;
                    scalar_t y_round_c = round(y) - y_center;
                    scalar_t z_round_c = round(z) - z_center;

                    scalar_t x_psf = r11 * x_round_c + r21 * y_round_c + r31 * z_round_c + (w_p-1)/2.;
                    scalar_t y_psf = r12 * x_round_c + r22 * y_round_c + r32 * z_round_c + (h_p-1)/2.;
                    scalar_t z_psf = r13 * x_round_c + r23 * y_round_c + r33 * z_round_c + (d_p-1)/2.;

                    if (x_psf < 0 || y_psf < 0 || z_psf < 0 || x_psf >= w_p-1 || y_psf >= h_p-1 || z_psf >= d_p-1) continue;

                    int32_t x_floor = floor(x_psf);
                    int32_t y_floor = floor(y_psf);
                    int32_t z_floor = floor(z_psf);
                    scalar_t wx = x_psf - x_floor;
                    scalar_t wy = y_psf - y_floor;
                    scalar_t wz = z_psf - z_floor;
                    int32_t i_v = z_floor * w_p * h_p + y_floor * w_p + x_floor;
                    psf_val = 0;
                    psf_val += (1-wx) * (1-wy) * (1-wz) * psf[i_v];
                    psf_val += wx * (1-wy) * (1-wz) * psf[i_v + 1];
                    psf_val += (1-wx) * wy * (1-wz) * psf[i_v + w_p];
                    psf_val += (1-wx) * (1-wy) * wz * psf[i_v + w_p*h_p];
                    psf_val += wx * wy * (1-wz) * psf[i_v + 1 + w_p];
                    psf_val += wx * (1-wy) * wz * psf[i_v + 1 + w_p*h_p];
                    psf_val += (1-wx) * wy * wz * psf[i_v + w_p + w_p*h_p];
                    psf_val += wx * wy * wz * psf[i_v + w_p + w_p*h_p + 1];

                } //else { // linear}
                weight += psf_val;
            }
        }
    }

    if (weight < 0.5) return; // border 
    //s /= weight;

    for (int iz_p = -d_p/2, i_p = 0; iz_p < (d_p+1)/2; iz_p++) {
        for (int iy_p = -h_p/2; iy_p < (h_p+1)/2; iy_p++) {
            for (int ix_p = -w_p/2; ix_p < (w_p+1)/2; ix_p++, i_p++) {
                scalar_t psf_val = psf[i_p];
                if (psf_val == 0) continue;
                scalar_t x = x_center + r11*ix_p + r12*iy_p + r13*iz_p;
                scalar_t y = y_center + r21*ix_p + r22*iy_p + r23*iz_p;
                scalar_t z = z_center + r31*ix_p + r32*iy_p + r33*iz_p;
                if (x < 0 || y < 0 || z < 0 || x >= W-1 || y >= H-1 || z >= D-1) continue;
                if (interp_psf) { // NN
                    int32_t x_round = round(x);
                    int32_t y_round = round(y);
                    int32_t z_round = round(z);

                    scalar_t x_psf = r11 * (x_round - x_center) + r21 * (y_round - y_center) + r31 * (z_round - z_center) + (w_p-1)/2.;
                    scalar_t y_psf = r12 * (x_round - x_center) + r22 * (y_round - y_center) + r32 * (z_round - z_center) + (h_p-1)/2.;
                    scalar_t z_psf = r13 * (x_round - x_center) + r23 * (y_round - y_center) + r33 * (z_round - z_center) + (d_p-1)/2.;

                    if (x_psf < 0 || y_psf < 0 || z_psf < 0 || x_psf >= w_p-1 || y_psf >= h_p-1 || z_psf >= d_p-1) continue;

                    int32_t x_floor = floor(x_psf);
                    int32_t y_floor = floor(y_psf);
                    int32_t z_floor = floor(z_psf);
                    scalar_t wx = x_psf - x_floor;
                    scalar_t wy = y_psf - y_floor;
                    scalar_t wz = z_psf - z_floor;
                    int32_t i_v = z_floor * w_p * h_p + y_floor * w_p + x_floor;
                    
                    psf_val = 0;
                    psf_val += (1-wx) * (1-wy) * (1-wz) * psf[i_v];
                    psf_val += wx * (1-wy) * (1-wz) * psf[i_v + 1];
                    psf_val += (1-wx) * wy * (1-wz) * psf[i_v + w_p];
                    psf_val += (1-wx) * (1-wy) * wz * psf[i_v +  w_p*h_p];
                    psf_val += wx * wy * (1-wz) * psf[i_v + 1 + w_p];
                    psf_val += wx * (1-wy) * wz * psf[i_v + 1 + w_p*h_p];
                    psf_val += (1-wx) * wy * wz * psf[i_v + w_p + w_p*h_p];
                    psf_val += wx * wy * wz * psf[i_v + w_p + w_p*h_p + 1];
                    psf_val /= weight;

                    i_v = z_round * Sz + y_round * Sy + x_round;
                    if (vol_mask != NULL && !vol_mask[i_v]) continue;
                    atomicAdd(vol + i_v, psf_val * s);
                    if (vol_weight != NULL) atomicAdd(vol_weight + i_v, psf_val);

                } else { // linear
                    int32_t x_floor = floor(x);
                    int32_t y_floor = floor(y);
                    int32_t z_floor = floor(z);
                    scalar_t wx = x - x_floor;
                    scalar_t wy = y - y_floor;
                    scalar_t wz = z - z_floor;
                    int32_t i_v = z_floor * Sz + y_floor * Sy + x_floor;

                    psf_val /= weight;
                    scalar_t psf_val_ = 0;

                    if (vol_mask == NULL || vol_mask[i_v]) {
                        psf_val_ = (1-wx) * (1-wy) * (1-wz) * psf_val;
                        atomicAdd(vol + i_v, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v, psf_val_);
                    }
                    
                    if (vol_mask == NULL || vol_mask[i_v + 1]) {
                        psf_val_ = wx * (1-wy) * (1-wz) * psf_val;
                        atomicAdd(vol + i_v + 1, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + 1, psf_val_);
                    }

                    if (vol_mask == NULL || vol_mask[i_v + Sy]) {
                        psf_val_ = (1-wx) * wy * (1-wz) * psf_val;
                        atomicAdd(vol + i_v + Sy, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + Sy, psf_val_);
                    }

                    if (vol_mask == NULL || vol_mask[i_v + Sz]) {
                        psf_val_ = (1-wx) * (1-wy) * wz * psf_val;
                        atomicAdd(vol + i_v + Sz, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + Sz, psf_val_);
                    }

                    if (vol_mask == NULL || vol_mask[i_v + 1 + Sy]) {
                        psf_val_ = wx * wy * (1-wz) * psf_val;
                        atomicAdd(vol + i_v + 1 + Sy, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + 1 + Sy, psf_val_);
                    }

                    if (vol_mask == NULL || vol_mask[i_v + 1 + Sz]) {
                        psf_val_ = wx * (1-wy) * wz * psf_val;
                        atomicAdd(vol + i_v + 1 + Sz, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + 1 + Sz, psf_val_);
                    }
                    
                    if (vol_mask == NULL || vol_mask[i_v + Sy + Sz]) {
                        psf_val_ = (1-wx) * wy * wz * psf_val;
                        atomicAdd(vol + i_v + Sy + Sz, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + Sy + Sz, psf_val_);
                    }

                    if (vol_mask == NULL || vol_mask[i_v + Sy + Sz + 1]) {
                        psf_val_ = wx * wy * wz *  psf_val;
                        atomicAdd(vol + i_v + Sy + Sz + 1, psf_val_ * s);
                        if (vol_weight != NULL) atomicAdd(vol_weight + i_v + Sy + Sz + 1, psf_val_);
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void equalize_cuda_kernel( 
    scalar_t* __restrict__ vol,
    const scalar_t* __restrict__ vol_weight,
    const bool is_grad,
    const int32_t DHW
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= DHW) return;
    const scalar_t weight = vol_weight[idx];
    if (weight > 0) {
        if (is_grad) {
            if (weight < 1e-3){
                vol[idx] /= 1e-3;
            } else {
                vol[idx] /= weight;
            }
        } else {
            vol[idx] /= weight;
        }
    }
}

template <typename scalar_t>
__global__ void slice_acquisition_adjoint_backward_cuda_kernel( 
    const scalar_t* __restrict__ transforms,
    const scalar_t* __restrict__ grad_vol,
    const scalar_t* __restrict__ psf,
    const scalar_t* __restrict__ slices,
    const bool* __restrict__ slices_mask,
    const scalar_t* __restrict__ vol,
    const bool* __restrict__ vol_mask,
    scalar_t* __restrict__ grad_slices,
    scalar_t* __restrict__ grad_transforms,
    const int32_t D, const int32_t H, const int32_t W,
    const int32_t d_p, const int32_t h_p, const int32_t w_p,
    const int32_t n, const int32_t h, const int32_t w,
    const scalar_t res_slice,
    const bool interp_psf
) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n*h*w) return;
    if (slices_mask != NULL && !slices_mask[idx]) return;
    const int32_t Sy = W, Sz = H*W;

    const int32_t ix = idx % w;
    const int32_t iy = (idx / w) % h;
    const int32_t in = idx / (h * w); 

    const scalar_t r11 = transforms[in*12  ], r12 = transforms[in*12+1], r13 = transforms[in*12+2];
    const scalar_t r21 = transforms[in*12+4], r22 = transforms[in*12+5], r23 = transforms[in*12+6];
    const scalar_t r31 = transforms[in*12+8], r32 = transforms[in*12+9], r33 = transforms[in*12+10];

    scalar_t _x = (ix - (w - 1) / 2.) * res_slice + transforms[in*12+3];
    scalar_t _y = (iy - (h - 1) / 2.) * res_slice + transforms[in*12+7];
    scalar_t _z = transforms[in*12+11];

    scalar_t x_center = r11 * _x + r12 * _y + r13 * _z;
    scalar_t y_center = r21 * _x + r22 * _y + r23 * _z;
    scalar_t z_center = r31 * _x + r32 * _y + r33 * _z;

    x_center += (W - 1) / 2.;
    y_center += (H - 1) / 2.;
    z_center += (D - 1) / 2.;

    scalar_t val = 0;
    scalar_t weight = 0;

    scalar_t g_r11 = 0, g_r12 = 0, g_r13 = 0, g_tx = 0;
    scalar_t g_r21 = 0, g_r22 = 0, g_r23 = 0, g_ty = 0;
    scalar_t g_r31 = 0, g_r32 = 0, g_r33 = 0, g_tz = 0;
   
    for (int iz_p = -d_p/2, i_p = 0; iz_p < (d_p+1)/2; iz_p++) {
        for (int iy_p = -h_p/2; iy_p < (h_p+1)/2; iy_p++) {
            for (int ix_p = -w_p/2; ix_p < (w_p+1)/2; ix_p++, i_p++) {
                scalar_t psf_val = psf[i_p];
                if (psf_val == 0) continue;
                scalar_t x = x_center + r11*ix_p + r12*iy_p + r13*iz_p;
                scalar_t y = y_center + r21*ix_p + r22*iy_p + r23*iz_p;
                scalar_t z = z_center + r31*ix_p + r32*iy_p + r33*iz_p;
                if (x < 0 || y < 0 || z < 0 || x >= W-1 || y >= H-1 || z >= D-1) continue;
                scalar_t val_ = 0;
                if (interp_psf) { // NN
                    int32_t x_round = round(x);
                    int32_t y_round = round(y);
                    int32_t z_round = round(z);
                    if (vol_mask != NULL && !vol_mask[z_round * Sz + y_round * Sy + x_round]) continue;
                    val_ = grad_vol[z_round * Sz + y_round * Sy + x_round];

                    scalar_t x_psf = r11 * (x_round - x_center) + r21 * (y_round - y_center) + r31 * (z_round - z_center) + (w_p-1)/2.;
                    scalar_t y_psf = r12 * (x_round - x_center) + r22 * (y_round - y_center) + r32 * (z_round - z_center) + (h_p-1)/2.;
                    scalar_t z_psf = r13 * (x_round - x_center) + r23 * (y_round - y_center) + r33 * (z_round - z_center) + (d_p-1)/2.;

                    if (x_psf < 0 || y_psf < 0 || z_psf < 0 || x_psf >= w_p-1 || y_psf >= h_p-1 || z_psf >= d_p-1) continue;

                    int32_t x_floor = floor(x_psf);
                    int32_t y_floor = floor(y_psf);
                    int32_t z_floor = floor(z_psf);
                    scalar_t wx = x_psf - x_floor;
                    scalar_t wy = y_psf - y_floor;
                    scalar_t wz = z_psf - z_floor;
                    int32_t i_v = z_floor * w_p * h_p + y_floor * w_p + x_floor;
                    
                    psf_val = 0;
                    psf_val += (1-wx) * (1-wy) * (1-wz) * psf[i_v];
                    psf_val += wx * (1-wy) * (1-wz) * psf[i_v + 1];
                    psf_val += (1-wx) * wy * (1-wz) * psf[i_v + w_p];
                    psf_val += (1-wx) * (1-wy) * wz * psf[i_v + w_p*h_p];
                    psf_val += wx * wy * (1-wz) * psf[i_v + 1 + w_p];
                    psf_val += wx * (1-wy) * wz * psf[i_v + 1 + w_p*h_p];
                    psf_val += (1-wx) * wy * wz * psf[i_v + w_p + w_p*h_p];
                    psf_val += wx * wy * wz * psf[i_v + w_p + w_p*h_p + 1];

                    if (grad_transforms != NULL) {
                        scalar_t dx = 0, dy = 0, dz = 0, s = 0;

                        s = psf[i_v];
                        dx -= (1-wy) * (1-wz) * s;
                        dy -= (1-wx) * (1-wz) * s;
                        dz -= (1-wx) * (1-wy) * s;

                        s = psf[i_v + 1];
                        dx += (1-wy) * (1-wz) * s;
                        dy -= wx * (1-wz) * s;
                        dz -= wx * (1-wy) * s;

                        s = psf[i_v + w_p];
                        dx -= wy * (1-wz) * s;
                        dy += (1-wx) * (1-wz) * s;
                        dz -= (1-wx) * wy * s;
                        
                        s = psf[i_v + w_p*h_p];
                        dx -= (1-wy) * wz * s;
                        dy -= (1-wx) * wz * s;
                        dz += (1-wx) * (1-wy) * s;

                        s = psf[i_v + 1 + w_p];
                        dx += wy * (1-wz) * s;
                        dy += wx * (1-wz) * s;
                        dz -= wx * wy * s;

                        s = psf[i_v + 1 + w_p*h_p];
                        dx += (1-wy) * wz * s;
                        dy -= wx * wz * s;
                        dz += wx * (1-wy) * s;

                        s = psf[i_v + w_p + w_p*h_p];
                        dx -= wy * wz * s;
                        dy += (1-wx) * wz * s;
                        dz += (1-wx) * wy * s;

                        s = psf[i_v + w_p + w_p*h_p + 1];
                        dx += wy * wz * s;
                        dy += wx * wz * s;
                        dz += wx * wy * s;

                        s = (vol == NULL) ? (slices[idx] * val_) : ((slices[idx] - vol[z_round * Sz + y_round * Sy + x_round]) * val_);
                        dx *= s;
                        dy *= s;
                        dz *= s;

                        g_r11 += dx * (x_round - (W - 1) / 2.), g_r12 += dy * (x_round - (W - 1) / 2.), g_r13 += dz * (x_round - (W - 1) / 2.);
                        g_r21 += dx * (y_round - (H - 1) / 2.), g_r22 += dy * (y_round - (H - 1) / 2.), g_r23 += dz * (y_round - (H - 1) / 2.);
                        g_r31 += dx * (z_round - (D - 1) / 2.), g_r32 += dy * (z_round - (D - 1) / 2.), g_r33 += dz * (z_round - (D - 1) / 2.);
                        g_tx -= dx;
                        g_ty -= dy;
                        g_tz -= dz;
                    }
                
                } else { // linear
                    
                    int32_t x_floor = floor(x);
                    int32_t y_floor = floor(y);
                    int32_t z_floor = floor(z);
                    scalar_t wx = x - x_floor;
                    scalar_t wy = y - y_floor;
                    scalar_t wz = z - z_floor;
                    int32_t i_v = z_floor * Sz + y_floor * Sy + x_floor;

                    if (grad_slices != NULL) {
                        if (vol_mask == NULL || vol_mask[i_v]) val_ += (1-wx) * (1-wy) * (1-wz) * grad_vol[i_v];
                        if (vol_mask == NULL || vol_mask[i_v+1]) val_ += wx * (1-wy) * (1-wz) * grad_vol[i_v + 1];
                        if (vol_mask == NULL || vol_mask[i_v+Sy]) val_ += (1-wx) * wy * (1-wz) * grad_vol[i_v + Sy];
                        if (vol_mask == NULL || vol_mask[i_v+Sz]) val_ += (1-wx) * (1-wy) * wz * grad_vol[i_v + Sz];
                        if (vol_mask == NULL || vol_mask[i_v+1+Sy]) val_ += wx * wy * (1-wz) * grad_vol[i_v + 1 + Sy];
                        if (vol_mask == NULL || vol_mask[i_v+1+Sz]) val_ += wx * (1-wy) * wz * grad_vol[i_v + 1 + Sz];
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz]) val_ += (1-wx) * wy * wz * grad_vol[i_v + Sy + Sz];
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz+1]) val_ += wx * wy * wz * grad_vol[i_v + Sy + Sz + 1];
                    }
                    
                    if (grad_transforms != NULL) {
                        scalar_t dx = 0, dy = 0, dz = 0, s = 0;
                        if (vol_mask == NULL || vol_mask[i_v]) {
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v]) : ((slices[idx] - vol[i_v]) * grad_vol[i_v]);
                            dx -= (1-wy) * (1-wz) * s;
                            dy -= (1-wx) * (1-wz) * s;
                            dz -= (1-wx) * (1-wy) * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+1]) {
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + 1]) : ((slices[idx] - vol[i_v + 1]) * grad_vol[i_v + 1]);
                            dx += (1-wy) * (1-wz) * s;
                            dy -= wx * (1-wz) * s;
                            dz -= wx * (1-wy) * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sy]) {
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + Sy]) : ((slices[idx] - vol[i_v + Sy]) * grad_vol[i_v + Sy]);
                            dx -= wy * (1-wz)  * s;
                            dy += (1-wx) * (1-wz) * s;
                            dz -= (1-wx) * wy  * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sz]) {
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + Sz]) : ((slices[idx] - vol[i_v + Sz]) * grad_vol[i_v + Sz]);
                            dx -= (1-wy) * wz * s;
                            dy -= (1-wx) * wz * s;
                            dz += (1-wx) * (1-wy) * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+1+Sy]) {
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + 1 + Sy]) : ((slices[idx] - vol[i_v + 1 + Sy]) * grad_vol[i_v + 1 + Sy]);
                            dx += wy * (1-wz) * s;
                            dy += wx * (1-wz) * s;
                            dz -= wx * wy * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+1+Sz]) { 
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + 1 + Sz]) : ((slices[idx] - vol[i_v + 1 + Sz]) * grad_vol[i_v + 1 + Sz]);
                            dx += (1-wy) * wz * s;
                            dy -= wx * wz * s;
                            dz += wx * (1-wy) * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz]) { 
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + Sy + Sz]) : ((slices[idx] - vol[i_v + Sy + Sz]) * grad_vol[i_v + Sy + Sz]);
                            dx -= wy * wz * s;
                            dy += (1-wx) * wz * s;
                            dz += (1-wx) * wy * s;
                        }
                        if (vol_mask == NULL || vol_mask[i_v+Sy+Sz+1]) {
                            s = (vol == NULL) ? (slices[idx] * grad_vol[i_v + Sy + Sz + 1]) : ((slices[idx] - vol[i_v + Sy + Sz + 1]) * grad_vol[i_v + Sy + Sz + 1]);
                            dx += wy * wz * s;
                            dy += wx * wz * s;
                            dz += wx * wy * s;
                        }

                        dx *= psf_val;
                        dy *= psf_val;
                        dz *= psf_val;

                        g_r11 += dx * (_x + ix_p), g_r12 += dx * (_y + iy_p), g_r13 += dx * (_z + iz_p);
                        g_r21 += dy * (_x + ix_p), g_r22 += dy * (_y + iy_p), g_r23 += dy * (_z + iz_p);
                        g_r31 += dz * (_x + ix_p), g_r32 += dz * (_y + iy_p), g_r33 += dz * (_z + iz_p);
                        g_tx += dx * r11 + dy * r21 + dz * r31;
                        g_ty += dx * r12 + dy * r22 + dz * r32;
                        g_tz += dx * r13 + dy * r23 + dz * r33;
                    }
                }
                val += psf_val * val_;
                weight += psf_val;
            }
        }
    }
    if (weight > 0) {
        if (grad_slices != NULL) {
            grad_slices[idx] = val / weight;
        }
        
        if (grad_transforms != NULL) {
            atomicAdd(grad_transforms + in * 12, g_r11 / weight);
            atomicAdd(grad_transforms + in * 12 + 1, g_r12 / weight);
            atomicAdd(grad_transforms + in * 12 + 2, g_r13 / weight);
            atomicAdd(grad_transforms + in * 12 + 3, g_tx / weight);
            atomicAdd(grad_transforms + in * 12 + 4, g_r21 / weight);
            atomicAdd(grad_transforms + in * 12 + 5, g_r22 / weight);
            atomicAdd(grad_transforms + in * 12 + 6, g_r23 / weight);
            atomicAdd(grad_transforms + in * 12 + 7, g_ty / weight);
            atomicAdd(grad_transforms + in * 12 + 8, g_r31 / weight);
            atomicAdd(grad_transforms + in * 12 + 9, g_r32 / weight);
            atomicAdd(grad_transforms + in * 12 + 10, g_r33 / weight);
            atomicAdd(grad_transforms + in * 12 + 11, g_tz / weight);
        }
    }
}

} // namespace

std::vector<torch::Tensor> slice_acquisition_forward_cuda(
    torch::Tensor transforms,
    torch::Tensor vol,
    torch::Tensor vol_mask,
    torch::Tensor slices_mask,
    torch::Tensor psf,
    torch::IntArrayRef slice_shape, // h,w
    const float res_slice,
    const bool need_weight,
    const bool interp_psf
) {
    auto slices = torch::zeros({transforms.size(0), 1, slice_shape[0], slice_shape[1]}, vol.options());
    auto slices_weight = need_weight ? torch::zeros_like(slices) : torch::Tensor();
    const int32_t threads = 512;
    const int32_t blocks = (slices.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(vol.type(), "slice_acquisition_forward_cuda", [&] {
        slice_acquisition_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            transforms.data_ptr<scalar_t>(),
            vol.data_ptr<scalar_t>(),
            vol_mask.numel() ? vol_mask.data_ptr<bool>() : NULL,
            psf.data_ptr<scalar_t>(),
            slices.data_ptr<scalar_t>(),
            slices_mask.numel() ? slices_mask.data_ptr<bool>() : NULL,
            need_weight ? slices_weight.data_ptr<scalar_t>() : NULL,
            vol.size(2), vol.size(3), vol.size(4),
            psf.size(0), psf.size(1), psf.size(2),
            slices.size(0), slices.size(2), slices.size(3),
            res_slice,
            interp_psf);
    });
    if (need_weight) {
        return {slices, slices_weight};
    }
    else {
        return {slices};
    }
}

std::vector<torch::Tensor> slice_acquisition_backward_cuda(
    torch::Tensor transforms,
    torch::Tensor vol,
    torch::Tensor vol_mask,
    torch::Tensor psf,
    torch::Tensor grad_slices,
    torch::Tensor slices_mask,
    const float res_slice,
    const bool interp_psf,
    const bool need_vol_grad,
    const bool need_transforms_grad
) {
    auto grad_vol = need_vol_grad ? torch::zeros_like(vol) : torch::Tensor();
    auto grad_transforms = need_transforms_grad ? torch::zeros_like(transforms) : torch::Tensor();
    const int32_t threads = 256;
    const int32_t blocks = (grad_slices.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(vol.type(), "slice_acquisition_backward_cuda", [&] {
        slice_acquisition_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            transforms.data_ptr<scalar_t>(),
            vol.data_ptr<scalar_t>(),
            vol_mask.numel() ? vol_mask.data_ptr<bool>() : NULL,
            psf.data_ptr<scalar_t>(),
            grad_slices.data_ptr<scalar_t>(),
            slices_mask.numel() ? slices_mask.data_ptr<bool>() : NULL,
            need_vol_grad ? grad_vol.data_ptr<scalar_t>() : NULL,
            need_transforms_grad ? grad_transforms.data_ptr<scalar_t>() : NULL,
            vol.size(2), vol.size(3), vol.size(4),
            psf.size(0), psf.size(1), psf.size(2),
            grad_slices.size(0), grad_slices.size(2), grad_slices.size(3),
            res_slice,
            interp_psf);
    });
    return {grad_vol, grad_transforms};
}

std::vector<torch::Tensor> slice_acquisition_adjoint_forward_cuda(
    torch::Tensor transforms,
    torch::Tensor psf,
    torch::Tensor slices,
    torch::Tensor slices_mask,
    torch::Tensor vol_mask,
    torch::IntArrayRef vol_shape, // D,H,W
    const float res_slice,
    const bool interp_psf,
    const bool equalize
) {
    auto vol = torch::zeros({1, 1, vol_shape[0], vol_shape[1], vol_shape[2]}, slices.options());
    auto vol_weight = equalize ? torch::zeros_like(vol) : torch::Tensor();
    
    int32_t threads = 256;
    int32_t blocks = (slices.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(vol.type(), "slice_acquisition_adjoint_forward_cuda", [&] {
        slice_acquisition_adjoint_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            transforms.data_ptr<scalar_t>(),
            vol.data_ptr<scalar_t>(),
            (equalize) ? vol_weight.data_ptr<scalar_t>() : NULL,
            vol_mask.numel() ? vol_mask.data_ptr<bool>() : NULL,
            psf.data_ptr<scalar_t>(),
            slices.data_ptr<scalar_t>(),
            slices_mask.numel() ? slices_mask.data_ptr<bool>() : NULL,
            vol.size(2), vol.size(3), vol.size(4),
            psf.size(0), psf.size(1), psf.size(2),
            slices.size(0), slices.size(2), slices.size(3),
            res_slice,
            interp_psf);
    });
    
    if (equalize) {
        threads = 1024;
        blocks = (vol.numel() + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(vol.type(), "equalize_cuda", [&] {
            equalize_cuda_kernel<scalar_t><<<blocks, threads>>>(
                vol.data_ptr<scalar_t>(),
                vol_weight.data_ptr<scalar_t>(),
                false,
                vol.size(2) * vol.size(3) * vol.size(4));
        });

    }

    return {vol, vol_weight};
}

std::vector<torch::Tensor> slice_acquisition_adjoint_backward_cuda(
    torch::Tensor transforms,
    torch::Tensor grad_vol,
    torch::Tensor vol_weight,
    torch::Tensor vol_mask,
    torch::Tensor psf,
    torch::Tensor slices,
    torch::Tensor slices_mask,
    torch::Tensor vol,
    const float res_slice,
    const bool interp_psf,
    const bool equalize,
    const bool need_slices_grad,
    const bool need_transforms_grad
) {
    
    if (equalize) {
        const int32_t threads = 1024;
        const int32_t blocks = (grad_vol.numel() + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(grad_vol.type(), "equalize_cuda", [&] {
            equalize_cuda_kernel<scalar_t><<<blocks, threads>>>(
                grad_vol.data_ptr<scalar_t>(),
                vol_weight.data_ptr<scalar_t>(),
                true,
                grad_vol.size(2) * grad_vol.size(3) * grad_vol.size(4));
        });

    }

    auto grad_slices = need_slices_grad ? torch::zeros_like(slices) : torch::Tensor();
    auto grad_transforms = need_transforms_grad ? torch::zeros_like(transforms) : torch::Tensor();
    const int32_t threads = 256;
    const int32_t blocks = (slices.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_vol.type(), "slice_acquisition_adjoint_backward_cuda", [&] {
        slice_acquisition_adjoint_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            transforms.data_ptr<scalar_t>(),
            grad_vol.data_ptr<scalar_t>(),
            psf.data_ptr<scalar_t>(),
            slices.data_ptr<scalar_t>(),
            slices_mask.numel() ? slices_mask.data_ptr<bool>() : NULL,
            equalize ? vol.data_ptr<scalar_t>() : NULL,
            vol_mask.numel() ? vol_mask.data_ptr<bool>() : NULL,
            need_slices_grad ? grad_slices.data_ptr<scalar_t>() : NULL,
            need_transforms_grad ? grad_transforms.data_ptr<scalar_t>() : NULL,
            grad_vol.size(2), grad_vol.size(3), grad_vol.size(4),
            psf.size(0), psf.size(1), psf.size(2),
            slices.size(0), slices.size(2), slices.size(3),
            res_slice,
            interp_psf);
    });
    return {grad_slices, grad_transforms};
}


