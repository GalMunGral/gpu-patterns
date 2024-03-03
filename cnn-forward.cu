#include <cmath>
#include <mma.h>
#include <iostream>
#include "cnn-forward.h"

using namespace nvcuda;

#define O_BASELINE 0    // Milestone 2 baseline
#define O_CONST_MEM 1   // Weight matrix (kernel values) in constant memory (0.5 point)
#define O_SHARED_MEM 2  // Tiled shared memory convolution (2 points)
#define O_MUL_UNROLL 3  // Shared memory matrix multiplication and input matrix unrolling (3 points)
#define O_KER_FUSION 4  // Kernel fusion for unrolling and matrix-multiplication (2 points)
#define O_TENSOR_CORE 5 // Using Tensor Cores to speed up matrix multiplication (5 points)
#define O_REG_TILED 6   // An advanced matrix multiplication algorithm (register-tiled) (5 points)
#define O_ADAPTIVE 7    // Multiple kernel implementations for different layer sizes (1 point)
#define O_STREAMS 8     // Using Streams to overlap computation with data transfer (4 points)

#define O_LEVEL O_ADAPTIVE

__host__ void checkErrors(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

/**************************************** kernels ****************************************/

#define N_STREAMS 4

#define T 64
#define U 16
#define L (T / U)

#define TILE_WIDTH 32
#define M_TILE_WIDTH 16
#define S_TILE_WIDTH 16

#define MAX_M 16
#define MAX_C 4
#define MAX_K 8
#define MAX_S 4

#define MAX_INPUT_TILE_WIDTH ((MAX_K) + (MAX_S) * ((S_TILE_WIDTH)-1))
#define MAX_INPUT_TILE_SIZE ((MAX_INPUT_TILE_WIDTH) * (MAX_INPUT_TILE_WIDTH))

__constant__ float kernel[MAX_M * MAX_C * MAX_K * MAX_K];

__global__ void conv_forward_baseline(float *__restrict__ output, const float *__restrict__ input, const float *__restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out - 1) / TILE_WIDTH + 1;
    const int b = blockIdx.z;
    const int m = blockIdx.x;
    const int by = blockIdx.y / W_grid;
    const int bx = blockIdx.y % W_grid;
    const int h = by * TILE_WIDTH + threadIdx.y;
    const int w = bx * TILE_WIDTH + threadIdx.x;

    if (h < H_out && w < W_out)
    {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c)
        {
            for (int p = 0; p < K; ++p)
            {
                for (int q = 0; q < K; ++q)
                {
                    acc += mask_4d(m, c, p, q) * in_4d(b, c, h * S + p, w * S + q);
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}

__global__ void conv_forward_const_mem(float *__restrict__ output, const float *__restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int W_grid = (W_out - 1) / TILE_WIDTH + 1;
    const int b = blockIdx.z;
    const int m = blockIdx.x;
    const int by = blockIdx.y / W_grid;
    const int bx = blockIdx.y % W_grid;
    const int h = by * TILE_WIDTH + threadIdx.y;
    const int w = bx * TILE_WIDTH + threadIdx.x;

    if (h < H_out && w < W_out)
    {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c)
        {
            for (int p = 0; p < K; ++p)
            {
                for (int q = 0; q < K; ++q)
                {
                    acc += mask_4d(m, c, p, q) * in_4d(b, c, h * S + p, w * S + q);
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}

__global__ void conv_forward_shared_mem(float *__restrict__ output, const float *__restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_grid = (W_out - 1) / S_TILE_WIDTH + 1;

    const int INPUT_TILE_WIDTH = K + S * (S_TILE_WIDTH - 1);
    const int INPUT_TILE_SIZE = INPUT_TILE_WIDTH * INPUT_TILE_WIDTH;

    __shared__ float input_tile[MAX_INPUT_TILE_SIZE];

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define tile_2d(i1, i0) input_tile[(i1) * (INPUT_TILE_WIDTH) + i0]

    const int b = blockIdx.z;
    const int m = blockIdx.x;
    const int by = blockIdx.y / W_grid;
    const int bx = blockIdx.y % W_grid;
    const int h_start = by * S_TILE_WIDTH;
    const int w_start = bx * S_TILE_WIDTH;
    const int input_h_start = h_start * S;
    const int input_w_start = w_start * S;
    const int h = h_start + threadIdx.y;
    const int w = w_start + threadIdx.x;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c)
    {
        int i = threadIdx.y * S_TILE_WIDTH + threadIdx.x;
        const int stride = S_TILE_WIDTH * S_TILE_WIDTH;
        while (i < INPUT_TILE_SIZE)
        {
            const int y = i / INPUT_TILE_WIDTH;
            const int x = i % INPUT_TILE_WIDTH;
            const int input_h = input_h_start + y;
            const int input_w = input_w_start + x;
            if (input_h < H && input_w < W)
            {
                tile_2d(y, x) = in_4d(b, c, input_h, input_w);
            }
            else
            {
                tile_2d(y, x) = 0.0f;
            }
            i += stride;
        }
        __syncthreads();
        for (int p = 0; p < K; ++p)
        {
            for (int q = 0; q < K; ++q)
            {
                const int y_start = threadIdx.y * S;
                const int x_start = threadIdx.x * S;
                acc += mask_4d(m, c, p, q) * tile_2d(y_start + p, x_start + q);
            }
        }
        __syncthreads();
    }

    if (h < H_out && w < W_out)
    {
        out_4d(b, m, h, w) = acc;
    }

#undef out_4d
#undef in_4d
#undef mask_4d
#undef tile_2d
}

__global__ void unroll_kernel(float *__restrict__ input_unrolled, const float *__restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_unrolled = C * K * K;
    const int W_unrolled = H_out * W_out;

#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define unrolled_3d(i2, i1, i0) input_unrolled[(i2) * (H_unrolled * W_unrolled) + (i1) * (W_unrolled) + i0]

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int row = by * blockDim.y + ty, col = bx * blockDim.x + tx;

    const int b = blockIdx.z;
    const int c = row / (K * K);
    const int p = row / K % K;
    const int q = row % K;
    const int h = col / W_out;
    const int w = col % W_out;

    if (row < H_unrolled && col < W_unrolled)
    {
        unrolled_3d(b, row, col) = in_4d(b, c, h * S + p, w * S + q);
    }

#undef in_4d
#undef unrolled_3d
}

__global__ void matmul_kernel(float *__restrict__ output, const float *__restrict__ input_unrolled, const int M, const int H_unrolled, const int W_unrolled)
{
#define in_3d(i2, i1, i0) input_unrolled[(i2) * (H_unrolled * W_unrolled) + (i1) * (W_unrolled) + i0]
#define out_3d(i2, i1, i0) output[(i2) * (M * W_unrolled) + (i1) * (W_unrolled) + i0]
#define mask_2d(i1, i0) kernel[(i1) * (H_unrolled) + i0]

    __shared__ float kernel_tile[M_TILE_WIDTH][M_TILE_WIDTH];
    __shared__ float input_tile[M_TILE_WIDTH][M_TILE_WIDTH];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int row = by * blockDim.y + ty, col = bx * blockDim.x + tx;
    const int b = blockIdx.z;

    const int numTiles = (H_unrolled - 1) / M_TILE_WIDTH + 1;
    float p = 0.0f;

    for (int i = 0; i < numTiles; ++i)
    {
        if (row < M && i * M_TILE_WIDTH + tx < H_unrolled)
        {
            kernel_tile[ty][tx] = mask_2d(row, i * M_TILE_WIDTH + tx);
        }
        else
        {
            kernel_tile[ty][tx] = 0.0f;
        }
        if (i * M_TILE_WIDTH + ty < H_unrolled && col < W_unrolled)
        {
            input_tile[ty][tx] = in_3d(b, i * M_TILE_WIDTH + ty, col);
        }
        else
        {
            input_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < M_TILE_WIDTH; ++j)
        {
            p += kernel_tile[ty][j] * input_tile[j][tx];
        }
        __syncthreads();
    }
    if (row < M && col < W_unrolled)
    {
        out_3d(b, row, col) = p;
    }

#undef mask_2d
#undef in_3d
#undef out_3d
}

__device__ float unroll(const float *input, const int b, const int row, const int col, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    const int W_out = (W - K) / S + 1;
    const int c = row / (K * K);
    const int p = row / K % K;
    const int q = row % K;
    const int h = col / W_out;
    const int w = col % W_out;
    return in_4d(b, c, h * S + p, w * S + q);

#undef in_4d
}

__global__ void conv_forward_matmul(float *__restrict__ output, const float *__restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_unrolled = C * K * K;
    const int W_unrolled = H_out * W_out;

#define out_3d(i2, i1, i0) output[(i2) * (M * W_unrolled) + (i1) * (W_unrolled) + i0]
#define mask_2d(i1, i0) kernel[(i1) * (H_unrolled) + i0]

    __shared__ float kernel_tile[M_TILE_WIDTH][M_TILE_WIDTH];
    __shared__ float input_tile[M_TILE_WIDTH][M_TILE_WIDTH];

    const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    const int row = by * blockDim.y + ty, col = bx * blockDim.x + tx;
    const int b = blockIdx.z;

    const int numTiles = (H_unrolled - 1) / M_TILE_WIDTH + 1;
    float p = 0.0f;

    for (int i = 0; i < numTiles; ++i)
    {
        if (row < M && i * M_TILE_WIDTH + tx < H_unrolled)
        {
            kernel_tile[ty][tx] = mask_2d(row, i * M_TILE_WIDTH + tx);
        }
        else
        {
            kernel_tile[ty][tx] = 0.0f;
        }
        if (i * M_TILE_WIDTH + ty < H_unrolled && col < W_unrolled)
        {
            input_tile[ty][tx] = unroll(input, b, i * M_TILE_WIDTH + ty, col, B, M, C, H, W, K, S);
        }
        else
        {
            input_tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < M_TILE_WIDTH; ++j)
        {
            p += kernel_tile[ty][j] * input_tile[j][tx];
        }
        __syncthreads();
    }
    if (row < M && col < W_unrolled)
    {
        out_3d(b, row, col) = p;
    }

#undef mask_2d
#undef out_3d
}

__global__ void conv_forward_reg_tiled(float *__restrict__ output, const float *__restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_unrolled = C * K * K;
    const int W_unrolled = H_out * W_out;

#define out_3d(i2, i1, i0) output[(i2) * (M * W_unrolled) + (i1) * (W_unrolled) + i0]

    __shared__ float kernel_tile[T];

    const int b = blockIdx.z;
    const int row_base = blockIdx.y * U;
    const int col = blockIdx.x * T + threadIdx.x;

    const int numTiles = (H_unrolled - 1) / L + 1;
    float P[U] = {0.0f};

    for (int i = 0; i < numTiles; ++i)
    {
        const int tx = threadIdx.x % L, ty = threadIdx.x / L;
        const int r = row_base + ty;
        const int c = i * L + tx;
        if (r < M && c < H_unrolled)
        {
            kernel_tile[ty * L + tx] = kernel[r * H_unrolled + c];
        }
        else
        {
            kernel_tile[ty * L + tx] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < L; ++j)
        {
            const int row = i * L + j;
            if (row < H_unrolled && col < W_unrolled)
            {
                const float val = unroll(input, b, row, col, B, M, C, H, W, K, S);
#pragma unroll
                for (int k = 0; k < U; ++k)
                {
                    P[k] += kernel_tile[k * L + j] * val;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < U; ++k)
    {
        const int row = row_base + k;
        if (row < M && col < W_unrolled)
        {
            out_3d(b, row, col) = P[k];
        }
    }

#undef out_3d
}

__global__ void conv_forward_tensor(float *__restrict__ output, const float *__restrict__ input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_unrolled = C * K * K;
    const int W_unrolled = H_out * W_out;

#define out_3d(i2, i1, i0) output[(i2) * (M * W_unrolled) + (i1) * (W_unrolled) + i0]
#define mask_2d(i1, i0) kernel[(i1) * (H_unrolled) + i0]

    __shared__ half kernel_tile[16][16];
    __shared__ half input_tile[16][16];
    __shared__ float output_tile[16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    const int b = blockIdx.z;
    const int row_start = blockIdx.y * 16;
    const int col_start = blockIdx.x * 16;
    const int y_start = threadIdx.x / 16;
    const int x = threadIdx.x % 16;

    const int numTiles = (H_unrolled - 1) / 16 + 1;

    wmma::fill_fragment(c_frag, 0.0f);
    for (int i = 0; i < numTiles; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int y = y_start + 2 * j;
            int row = row_start + y;
            int col = i * 16 + x;
            if (row < M && col < H_unrolled)
            {
                kernel_tile[y][x] = mask_2d(row, col);
            }
            else
            {
                kernel_tile[y][x] = 0.0f;
            }
        }
        for (int j = 0; j < 8; ++j)
        {
            int y = y_start + 2 * j;
            int row = i * 16 + y;
            int col = col_start + x;
            if (row < H_unrolled && col < W_unrolled)
            {
                input_tile[y][x] = unroll(input, b, row, col, B, M, C, H, W, K, S);
            }
            else
            {
                input_tile[y][x] = 0.0f;
            }
        }
        wmma::load_matrix_sync(a_frag, (half *)kernel_tile, 16);
        wmma::load_matrix_sync(b_frag, (half *)input_tile, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync((float *)output_tile, c_frag, 16, wmma::mem_row_major);
    for (int j = 0; j < 8; ++j)
    {
        int y = y_start + 2 * j;
        int row = row_start + y;
        int col = col_start + x;
        if (row < M && col < W_unrolled)
        {
            out_3d(b, row, col) = output_tile[y][x];
        }
    }

#undef mask_2d
#undef out_3d
}

/**************************************** device ptrs ****************************************/

float *input_unrolled;

cudaStream_t streams[N_STREAMS];

/**************************************** prolog ****************************************/

__host__ void baseline_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_mask_ptr, M * C * K * K * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void const_mem_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void shared_mem_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void matmul_unroll_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));
    checkErrors(cudaMalloc((void **)&input_unrolled, B * C * K * K * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void kernel_fusion_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void reg_tiled_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void adaptive_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void multi_stream_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int InSegSize = C * H * W;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));

    for (int i = 0; i < N_STREAMS; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < B; ++i)
    {
        checkErrors(cudaMemcpyAsync(*device_input_ptr + i * InSegSize, host_input + i * InSegSize, InSegSize * sizeof(float), cudaMemcpyHostToDevice, streams[i % N_STREAMS]));
    }
}

__host__ void tensor_prolog(const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    checkErrors(cudaMalloc((void **)device_input_ptr, B * C * H * W * sizeof(float)));
    checkErrors(cudaMalloc((void **)device_output_ptr, B * M * H_out * W_out * sizeof(float)));

    checkErrors(cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    checkErrors(cudaMemcpyToSymbol(kernel, host_mask, M * C * K * K * sizeof(float)));
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    switch (O_LEVEL)
    {
    case O_BASELINE:
        baseline_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, device_mask_ptr, B, M, C, H, W, K, S);
        break;
    case O_CONST_MEM:
        const_mem_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_SHARED_MEM:
        shared_mem_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_MUL_UNROLL:
        matmul_unroll_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_KER_FUSION:
        kernel_fusion_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_REG_TILED:
        reg_tiled_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_ADAPTIVE:
        adaptive_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_STREAMS:
        multi_stream_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    case O_TENSOR_CORE:
        tensor_prolog(host_input, host_mask, device_output_ptr, device_input_ptr, B, M, C, H, W, K, S);
        break;
    }
}

/**************************************** kernel launch ****************************************/

__host__ void baseline(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_grid = (H_out - 1) / TILE_WIDTH + 1;
    const int W_grid = (W_out - 1) / TILE_WIDTH + 1;
    const int Y = H_grid * W_grid;

    dim3 GridDim(M, Y, B);
    dim3 BlockDim(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_baseline<<<GridDim, BlockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
}

__host__ void const_mem(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_grid = (H_out - 1) / TILE_WIDTH + 1;
    const int W_grid = (W_out - 1) / TILE_WIDTH + 1;
    const int Y = H_grid * W_grid;

    dim3 GridDim(M, Y, B);
    dim3 BlockDim(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_const_mem<<<GridDim, BlockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
}

__host__ void shared_mem(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S, cudaStream_t stream = 0)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int H_grid = (H_out - 1) / S_TILE_WIDTH + 1;
    const int W_grid = (W_out - 1) / S_TILE_WIDTH + 1;
    const int Y = H_grid * W_grid;

    dim3 GridDim(M, Y, B);
    dim3 BlockDim(S_TILE_WIDTH, S_TILE_WIDTH, 1);

    conv_forward_shared_mem<<<GridDim, BlockDim, 0, stream>>>(device_output, device_input, B, M, C, H, W, K, S);
}

__host__ void matmul_unroll(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_unrolled = H_out * W_out;
    const int H_unrolled = C * K * K;

    dim3 BlockDim(M_TILE_WIDTH, M_TILE_WIDTH, 1);
    dim3 GridDim_unroll((W_unrolled - 1) / M_TILE_WIDTH + 1, (H_unrolled - 1) / M_TILE_WIDTH + 1, B);
    dim3 GridDim_matmul((W_unrolled - 1) / M_TILE_WIDTH + 1, (M - 1) / M_TILE_WIDTH + 1, B);

    unroll_kernel<<<GridDim_unroll, BlockDim>>>(input_unrolled, device_input, B, M, C, H, W, K, S);
    matmul_kernel<<<GridDim_matmul, BlockDim>>>(device_output, input_unrolled, M, H_unrolled, W_unrolled);
}

__host__ void kernel_fusion(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_unrolled = H_out * W_out;

    dim3 BlockDim(M_TILE_WIDTH, M_TILE_WIDTH, 1);
    dim3 GridDim((W_unrolled - 1) / M_TILE_WIDTH + 1, (M - 1) / M_TILE_WIDTH + 1, B);

    conv_forward_matmul<<<GridDim, BlockDim>>>(device_output, device_input, B, M, C, H, W, K, S);
}

__host__ void reg_tiled(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S, cudaStream_t stream = 0)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_unrolled = H_out * W_out;

    dim3 GridDim((W_unrolled - 1) / T + 1, (M - 1) / U + 1, B);

    conv_forward_reg_tiled<<<GridDim, T, 0, stream>>>(device_output, device_input, B, M, C, H, W, K, S);
}

__host__ void adaptive(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S, cudaStream_t stream = 0)
{
    if (M < 8)
    { // TODO
        shared_mem(device_output, device_input, B, M, C, H, W, K, S, stream);
    }
    else
    {
        reg_tiled(device_output, device_input, B, M, C, H, W, K, S, stream);
    }
}

__host__ void multi_stream(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int InSegSize = C * H * W;
    const int OutSegSize = M * H_out * W_out;

    for (int i = 0; i < B; ++i)
    {
        adaptive(device_output + i * OutSegSize, device_input + i * InSegSize, 1, M, C, H, W, K, S, streams[i % N_STREAMS]);
    }
}

__host__ void tensor(float *device_output, const float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int W_unrolled = H_out * W_out;

    dim3 GridDim((W_unrolled - 1) / 16 + 1, (M - 1) / 16 + 1, B);
    conv_forward_tensor<<<GridDim, 32>>>(device_output, device_input, B, M, C, H, W, K, S);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    switch (O_LEVEL)
    {
    case O_BASELINE:
        baseline(device_output, device_input, device_mask, B, M, C, H, W, K, S);
        break;
    case O_CONST_MEM:
        const_mem(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_SHARED_MEM:
        shared_mem(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_MUL_UNROLL:
        matmul_unroll(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_KER_FUSION:
        kernel_fusion(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_REG_TILED:
        reg_tiled(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_ADAPTIVE:
        adaptive(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_STREAMS:
        multi_stream(device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_TENSOR_CORE:
        tensor(device_output, device_input, B, M, C, H, W, K, S);
        break;
    }
}

/**************************************** epilog ****************************************/

__host__ void baseline_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
    checkErrors(cudaFree(device_mask));
}

__host__ void const_mem_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void shared_mem_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void matmul_unroll_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
    checkErrors(cudaFree(input_unrolled));
}

__host__ void kernel_fusion_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void reg_tiled_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void adaptive_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void multi_stream_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    const int InSegSize = C * H * W;
    const int OutSegSize = M * H_out * W_out;

    for (int i = 0; i < B; ++i)
    {
        checkErrors(cudaMemcpyAsync(host_output + i * OutSegSize, device_output + i * OutSegSize, OutSegSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i % N_STREAMS]));
    }

    for (int i = 0; i < N_STREAMS; ++i)
    {
        checkErrors(cudaStreamDestroy(streams[i]));
    }

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void tensor_epilog(float *host_output, float *device_output, float *device_input, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;
    checkErrors(cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost));

    checkErrors(cudaFree(device_output));
    checkErrors(cudaFree(device_input));
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    switch (O_LEVEL)
    {
    case O_BASELINE:
        baseline_epilog(host_output, device_output, device_input, device_mask, B, M, C, H, W, K, S);
        break;
    case O_CONST_MEM:
        const_mem_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_SHARED_MEM:
        shared_mem_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_MUL_UNROLL:
        matmul_unroll_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_KER_FUSION:
        kernel_fusion_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_REG_TILED:
        reg_tiled_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_ADAPTIVE:
        adaptive_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_STREAMS:
        multi_stream_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    case O_TENSOR_CORE:
        tensor_epilog(host_output, device_output, device_input, B, M, C, H, W, K, S);
        break;
    }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
