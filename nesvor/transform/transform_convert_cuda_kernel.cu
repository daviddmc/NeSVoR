#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

#define TRANSFORM_EPS 1e-6
#define DEGREE2RAD 0.01745329251F
#define RAD2DEGREE 57.2957795131F

template <typename scalar_t>
__global__ void axisangle2mat_forward_cuda_kernel( 
    scalar_t* __restrict__ mat, // nx3x4
    const scalar_t* __restrict__ axisangle, // nx6
    const int32_t n
){
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int32_t offset_m = idx * 12;
    const int32_t offset_a = idx * 6;

    scalar_t ang_x = axisangle[offset_a];
    scalar_t ang_y = axisangle[offset_a + 1];
    scalar_t ang_z = axisangle[offset_a + 2];
    #ifdef DEGREE
    ang_x *= DEGREE2RAD
    ang_y *= DEGREE2RAD
    ang_z *= DEGREE2RAD
    #endif

    const scalar_t theta2 = ang_x * ang_x + ang_y * ang_y + ang_z * ang_z;

    if (theta2 > TRANSFORM_EPS) {

        const scalar_t theta = sqrtf(theta2);
        ang_x /= theta; //+ TRANSFORM_EPS;
        ang_y /= theta; //+ TRANSFORM_EPS;
        ang_z /= theta; //+ TRANSFORM_EPS;
        const scalar_t s = sinf(theta);
        const scalar_t c = cosf(theta);
        const scalar_t o_c = 1 - c;

        mat[offset_m] = c + ang_x * ang_x * o_c; 
        mat[offset_m + 1] = ang_x * ang_y * o_c - ang_z * s;
        mat[offset_m + 2] = ang_y * s + ang_x * ang_z * o_c;
        mat[offset_m + 4] = ang_z * s + ang_x * ang_y * o_c;
        mat[offset_m + 5] = c + ang_y * ang_y * o_c; 
        mat[offset_m + 6] = -ang_x * s + ang_y * ang_z * o_c;
        mat[offset_m + 8] = -ang_y * s + ang_x * ang_z * o_c;
        mat[offset_m + 9] = ang_x * s + ang_y * ang_z * o_c;
        mat[offset_m + 10] = c + ang_z * ang_z * o_c;

    } else {
        mat[offset_m] = 1; mat[offset_m + 1] = -ang_z; mat[offset_m + 2] = ang_y;
        mat[offset_m + 4] = ang_z; mat[offset_m + 5] = 1; mat[offset_m + 6] = -ang_x;
        mat[offset_m + 8] = -ang_y; mat[offset_m + 9] = ang_x; mat[offset_m + 10] = 1;
    }

    mat[offset_m + 3] = axisangle[offset_a + 3];
    mat[offset_m + 7] = axisangle[offset_a + 4];
    mat[offset_m + 11] = axisangle[offset_a + 5];
}


template <typename scalar_t>
__global__ void axisangle2mat_backward_cuda_kernel( 
    const scalar_t* __restrict__ grad_mat, // nx3x4
    const scalar_t* __restrict__ axisangle, // nx6
    scalar_t* __restrict__ grad_axisangle, // nx6
    const int32_t n
){
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int32_t offset_m = idx * 12;
    const int32_t offset_a = idx * 6;

    scalar_t ang_x = axisangle[offset_a];
    scalar_t ang_y = axisangle[offset_a + 1];
    scalar_t ang_z = axisangle[offset_a + 2];

    const scalar_t theta2 = ang_x * ang_x + ang_y * ang_y + ang_z * ang_z;

    if (theta2 > TRANSFORM_EPS) {

        const scalar_t theta = sqrtf(theta2);
        ang_x /= theta;// + TRANSFORM_EPS;
        ang_y /= theta;// + TRANSFORM_EPS;
        ang_z /= theta;// + TRANSFORM_EPS;
        const scalar_t s = sinf(theta);
        const scalar_t c = cosf(theta);
        const scalar_t o_c = 1 - c;

        scalar_t dx = 0, dy = 0, dz = 0, ds = 0, dc = 0, g = 0;

        g = grad_mat[offset_m];
        dc += (1 - ang_x * ang_x) * g;
        dx += 2 * o_c * ang_x * g;
        
        g = grad_mat[offset_m + 1];
        dc -= ang_x * ang_y * g;
        ds -= ang_z * g;
        dx += ang_y * o_c * g;
        dy += ang_x * o_c * g;
        dz -= s * g;

        g = grad_mat[offset_m + 2];
        dc -= ang_x * ang_z * g;
        ds += ang_y * g;
        dx += ang_z * o_c * g;
        dy += s * g;
        dz += ang_x * o_c * g;

        g = grad_mat[offset_m + 4];
        dc -= ang_x * ang_y * g;
        ds += ang_z * g;
        dx += ang_y * o_c * g;
        dy += ang_x * o_c * g;
        dz += s * g;

        g = grad_mat[offset_m + 5];
        dc += (1 - ang_y * ang_y) * g;
        dy += 2 * o_c * ang_y * g;

        g = grad_mat[offset_m + 6];
        dc -= ang_y * ang_z * g;
        ds -= ang_x * g;
        dx -= s * g;
        dy += ang_z * o_c * g;
        dz += ang_y * o_c * g;

        g = grad_mat[offset_m + 8];
        dc -= ang_x * ang_z * g;
        ds -= ang_y * g;
        dx += ang_z * o_c * g;
        dy -= s * g;
        dz += ang_x * o_c * g;

        g = grad_mat[offset_m + 9];
        dc -= ang_y * ang_z * g;
        ds += ang_x * g;
        dx += s * g;
        dy += ang_z * o_c * g;
        dz += ang_y * o_c * g;

        g = grad_mat[offset_m + 10];
        dc += (1 - ang_z * ang_z) * g;
        dz += 2 * o_c * ang_z * g;

        g = (c * ds - s * dc) * ang_x;
        g += (dx * (1 - ang_x * ang_x) - (dy * ang_y + dz * ang_z) * ang_x) / theta; //(theta + TRANSFORM_EPS);
        #ifdef DEGREE
        g *= DEGREE2RAD
        #endif
        grad_axisangle[offset_a] = g;

        g = (c * ds - s * dc) * ang_y;
        g += (dy * (1 - ang_y * ang_y) - (dx * ang_x + dz * ang_z) * ang_y) / theta; //(theta + TRANSFORM_EPS);
        #ifdef DEGREE
        g *= DEGREE2RAD
        #endif
        grad_axisangle[offset_a + 1] = g;

        g = (c * ds - s * dc) * ang_z;
        g += (dz * (1 - ang_z * ang_z) - (dx * ang_x + dy * ang_y) * ang_z) / theta; //(theta + TRANSFORM_EPS);
        #ifdef DEGREE
        g *= DEGREE2RAD
        #endif
        grad_axisangle[offset_a + 2] = g;

    } else {
        #ifdef DEGREE
        grad_axisangle[offset_a] = DEGREE2RAD * (grad_mat[offset_m + 9] - grad_mat[offset_m + 6]);
        grad_axisangle[offset_a + 1] = DEGREE2RAD * (grad_mat[offset_m + 2] - grad_mat[offset_m + 8]);
        grad_axisangle[offset_a + 2] = DEGREE2RAD * (grad_mat[offset_m + 4] - grad_mat[offset_m + 1]);
        #else
        grad_axisangle[offset_a] = grad_mat[offset_m + 9] - grad_mat[offset_m + 6];
        grad_axisangle[offset_a + 1] = grad_mat[offset_m + 2] - grad_mat[offset_m + 8];
        grad_axisangle[offset_a + 2] = grad_mat[offset_m + 4] - grad_mat[offset_m + 1];
        #endif
    }

    grad_axisangle[offset_a + 3] = grad_mat[offset_m + 3];
    grad_axisangle[offset_a + 4] = grad_mat[offset_m + 7];
    grad_axisangle[offset_a + 5] = grad_mat[offset_m + 11];
}

template <typename scalar_t>
__global__ void mat2axisangle_forward_cuda_kernel( 
    const scalar_t* __restrict__ mat, // nx3x4
    scalar_t* __restrict__ axisangle, // nx6
    const int32_t n
){
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int32_t offset_m = idx * 12;
    const int32_t offset_a = idx * 6;

    const scalar_t r00 = mat[offset_m];
    const scalar_t r01 = mat[offset_m+1];
    const scalar_t r02 = mat[offset_m+2];
    const scalar_t r10 = mat[offset_m+4];
    const scalar_t r11 = mat[offset_m+5];
    const scalar_t r12 = mat[offset_m+6];
    const scalar_t r20 = mat[offset_m+8];
    const scalar_t r21 = mat[offset_m+9];
    const scalar_t r22 = mat[offset_m+10];

    bool mask_d2 = r22 < TRANSFORM_EPS;
    bool mask_d0_d1 = r00 > r11;
    bool mask_d0_nd1 = r00 < -r11;
    
    scalar_t w = 0, x = 0, y = 0, z = 0;

    if (!mask_d2 && !mask_d0_nd1) {
        scalar_t s = 2 * sqrtf(r00 + r11 + r22 + 1);
        w = 0.25 * s;
        x = (r21 - r12) / s;
        y = (r02 - r20) / s;
        z = (r10 - r01) / s;
    } else if (mask_d2 && mask_d0_d1) {
        scalar_t s = 2 * sqrtf(r00 - r11 - r22 + 1);
        w = (r21 - r12) / s;
        x = 0.25 * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
    } else if (mask_d2 && !mask_d0_d1) {
        scalar_t s = 2 * sqrtf(r11 - r00 - r22 + 1);
        w = (r02 - r20) / s;
        x = (r01 + r10) / s;
        y = 0.25 * s;
        z = (r12 + r21) / s;
    } else {
        scalar_t s = 2 * sqrtf(r22 - r00 - r11 + 1);
        w = (r10 - r01) / s;
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25 * s;
    }
    if (w < 0) {
        w *= -1;
        x *= -1;
        y *= -1;
        z *= -1;
    }
    scalar_t tmp = x*x + y*y + z*z;
    scalar_t si = sqrtf(tmp); 
    scalar_t theta = 2 * atan2f(si, w);
    scalar_t fac = (tmp > TRANSFORM_EPS) ? (theta / si) : 2.0 / w;
    #ifdef DEGREE
    axisangle[offset_a] = x * fac * RAD2DEGREE;
    axisangle[offset_a + 1] = y * fac * RAD2DEGREE;
    axisangle[offset_a + 2] = z * fac * RAD2DEGREE;
    #else
    axisangle[offset_a] = x * fac;
    axisangle[offset_a + 1] = y * fac;
    axisangle[offset_a + 2] = z * fac;
    #endif
    axisangle[offset_a + 3] = mat[offset_m + 3];
    axisangle[offset_a + 4] = mat[offset_m + 7];
    axisangle[offset_a + 5] = mat[offset_m + 11];
}

template <typename scalar_t>
__global__ void mat2axisangle_backward_cuda_kernel( 
    const scalar_t* __restrict__ mat, // nx3x4
    const scalar_t* __restrict__ grad_axisangle, // nx6
    scalar_t* __restrict__ grad_mat, // nx3x4
    const int32_t n
){
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int32_t offset_m = idx * 12;
    const int32_t offset_a = idx * 6;

    const scalar_t r00 = mat[offset_m];
    const scalar_t r01 = mat[offset_m+1];
    const scalar_t r02 = mat[offset_m+2];
    const scalar_t r10 = mat[offset_m+4];
    const scalar_t r11 = mat[offset_m+5];
    const scalar_t r12 = mat[offset_m+6];
    const scalar_t r20 = mat[offset_m+8];
    const scalar_t r21 = mat[offset_m+9];
    const scalar_t r22 = mat[offset_m+10];

    bool mask_d2 = r22 < TRANSFORM_EPS;
    bool mask_d0_d1 = r00 > r11;
    bool mask_d0_nd1 = r00 < -r11;
    
    scalar_t w = 0, x = 0, y = 0, z = 0, s = 0;

    if (!mask_d2 && !mask_d0_nd1) {
        s = 2 * sqrtf(r00 + r11 + r22 + 1);
        w = 0.25 * s;
        x = (r21 - r12) / s;
        y = (r02 - r20) / s;
        z = (r10 - r01) / s;
    } else if (mask_d2 && mask_d0_d1) {
        s = 2 * sqrtf(r00 - r11 - r22 + 1);
        w = (r21 - r12) / s;
        x = 0.25 * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
    } else if (mask_d2 && !mask_d0_d1) {
        s = 2 * sqrtf(r11 - r00 - r22 + 1);
        w = (r02 - r20) / s;
        x = (r01 + r10) / s;
        y = 0.25 * s;
        z = (r12 + r21) / s;
    } else {
        s = 2 * sqrtf(r22 - r00 - r11 + 1);
        w = (r10 - r01) / s;
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25 * s;
    }

    bool neg_w = w < 0;
    if (neg_w) {
        w *= -1;
        x *= -1;
        y *= -1;
        z *= -1;
    }

    scalar_t tmp = x*x + y*y + z*z;
    const scalar_t si = sqrtf(tmp);
    const scalar_t theta = 2 * atan2f(si, w);

    scalar_t dw = x * grad_axisangle[offset_a] + y * grad_axisangle[offset_a+1] + z * grad_axisangle[offset_a+2];
    #ifdef DEGREE
    dw *= RAD2DEGREE;
    #endif
    scalar_t dx = dw;
    scalar_t dy = dw;
    scalar_t dz = dw;

    scalar_t fac;
    if (tmp > TRANSFORM_EPS) {
        fac = theta / si;
        tmp = 2 / (w*w + si*si);
        dw *= -tmp;
        tmp = (w * tmp - fac) / si;
        dx *= tmp * (x / si);
        dy *= tmp * (y / si);
        dz *= tmp * (z / si);
    } else {
        fac = 2.0 / w;
        tmp = 2 / (w*w + si*si);
        dw *= -tmp;
        tmp = (w * tmp - fac) / (si + TRANSFORM_EPS);
        dx *= tmp * (x / (si + TRANSFORM_EPS));
        dy *= tmp * (y / (si + TRANSFORM_EPS));
        dz *= tmp * (z / (si + TRANSFORM_EPS));
    }
    #ifdef DEGREE
    fac *= RAD2DEGREE;
    #endif
    dx += fac * grad_axisangle[offset_a];
    dy += fac * grad_axisangle[offset_a+1];
    dz += fac * grad_axisangle[offset_a+2];

    if (neg_w) {
        w *= -1;
        x *= -1;
        y *= -1;
        z *= -1;
        dx *= -1;
        dy *= -1;
        dz *= -1;
        dw *= -1;
    }

    if (!mask_d2 && !mask_d0_nd1) {
        
        grad_mat[offset_m+9] = dx / s;  // r21
        grad_mat[offset_m+6] = -dx / s; // r12
        grad_mat[offset_m+2] = dy / s;  // r02
        grad_mat[offset_m+8] = -dy / s; // r20
        grad_mat[offset_m+4] = dz / s;  // r10
        grad_mat[offset_m+1] = -dz / s; // r01

        scalar_t ds = - (x * dx + y * dy + z * dz) / s + 0.25 * dw;
        ds *= 2 / s;
        grad_mat[offset_m] = ds;
        grad_mat[offset_m+5] = ds;
        grad_mat[offset_m+10] = ds;

    } else if (mask_d2 && mask_d0_d1) {

        grad_mat[offset_m+9] = dw / s;  // r21
        grad_mat[offset_m+6] = -dw / s; // r12
        grad_mat[offset_m+2] = dz / s;  // r02
        grad_mat[offset_m+8] = dz / s; // r20
        grad_mat[offset_m+4] = dy / s;  // r10
        grad_mat[offset_m+1] = dy / s; // r01

        scalar_t ds = - (w * dw + y * dy + z * dz) / s + 0.25 * dx;
        ds *= 2 / s;
        grad_mat[offset_m] = ds;
        grad_mat[offset_m+5] = -ds;
        grad_mat[offset_m+10] = -ds;

    } else if (mask_d2 && !mask_d0_d1) {

        grad_mat[offset_m+9] = dz / s;  // r21
        grad_mat[offset_m+6] = dz / s; // r12
        grad_mat[offset_m+2] = dw / s;  // r02
        grad_mat[offset_m+8] = -dw / s; // r20
        grad_mat[offset_m+4] = dx / s;  // r10
        grad_mat[offset_m+1] = dx / s; // r01

        scalar_t ds = - (x * dx + w * dw + z * dz) / s + 0.25 * dy;
        ds *= 2 / s;
        grad_mat[offset_m] = -ds;
        grad_mat[offset_m+5] = ds;
        grad_mat[offset_m+10] = -ds;
        
    } else {

        grad_mat[offset_m+9] = dy / s;  // r21
        grad_mat[offset_m+6] = dy / s; // r12
        grad_mat[offset_m+2] = dx / s;  // r02
        grad_mat[offset_m+8] = dx / s; // r20
        grad_mat[offset_m+4] = dw / s;  // r10
        grad_mat[offset_m+1] = -dw / s; // r01

        scalar_t ds = - (x * dx + y * dy + w * dw) / s + 0.25 * dz;
        ds *= 2 / s;
        grad_mat[offset_m] = -ds;
        grad_mat[offset_m+5] = -ds;
        grad_mat[offset_m+10] = ds;
    }
    
    grad_mat[offset_m + 3] = grad_axisangle[offset_a + 3];
    grad_mat[offset_m + 7] = grad_axisangle[offset_a + 4];
    grad_mat[offset_m + 11] = grad_axisangle[offset_a + 5];
}

} // namespace

std::vector<torch::Tensor> axisangle2mat_forward_cuda(
    torch::Tensor axisangle
) {
    auto mat = torch::zeros({axisangle.size(0), 3, 4}, axisangle.options());
    const int32_t threads = 256;
    const int32_t blocks = (axisangle.size(0) + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(axisangle.type(), "axisangle2mat_forward_cuda", [&] {
        axisangle2mat_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            mat.data_ptr<scalar_t>(),
            axisangle.data_ptr<scalar_t>(),
            axisangle.size(0));
    });

    return {mat};
}

std::vector<torch::Tensor> axisangle2mat_backward_cuda(
    torch::Tensor grad_mat,
    torch::Tensor axisangle
) {
    auto grad_axisangle = torch::zeros_like(axisangle);
    const int32_t threads = 256;
    const int32_t blocks = (axisangle.size(0) + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(axisangle.type(), "axisangle2mat_backward_cuda", [&] {
        axisangle2mat_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_mat.data_ptr<scalar_t>(),
            axisangle.data_ptr<scalar_t>(),
            grad_axisangle.data_ptr<scalar_t>(),
            axisangle.size(0));
    });

    return {grad_axisangle};
}

std::vector<torch::Tensor> mat2axisangle_forward_cuda(
    torch::Tensor mat
) {
    auto axisangle = torch::zeros({mat.size(0), 6}, mat.options());
    const int32_t threads = 256;
    const int32_t blocks = (mat.size(0) + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(mat.type(), "mat2axisangle_forward_cuda", [&] {
        mat2axisangle_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            mat.data_ptr<scalar_t>(),
            axisangle.data_ptr<scalar_t>(),
            axisangle.size(0));
    });

    return {axisangle};
}

std::vector<torch::Tensor> mat2axisangle_backward_cuda(
    torch::Tensor mat,
    torch::Tensor grad_axisangle
) {
    auto grad_mat = torch::zeros_like(mat);
    const int32_t threads = 256;
    const int32_t blocks = (mat.size(0) + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(mat.type(), "mat2axisangle_backward_cuda", [&] {
        mat2axisangle_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            mat.data_ptr<scalar_t>(),
            grad_axisangle.data_ptr<scalar_t>(),
            grad_mat.data_ptr<scalar_t>(),
            mat.size(0));
    });

    return {grad_mat};
}