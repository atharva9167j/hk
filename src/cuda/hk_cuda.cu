// HK CUDA backend: minimal device-side GEMV kernels used by the native Zig
// engine when built with `-Dcuda=true`. Exposes a small extern "C" ABI so
// src/cuda.zig can bind to it without any C++ interop machinery.
//
// Scope (first milestone): host<->device transfer + correct GEMV for the
// storage types the native engine already supports on CPU (F32, Q8_0).
// This is NOT a full CUDA port of TransformerEngine.forward() yet -
// attention/RoPE/KV-cache still run on CPU; only the matVec step can be
// routed to the GPU so far.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {

int hk_cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

int hk_cuda_get_device_name(char* buf, int buf_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return (int)err;
    std::strncpy(buf, prop.name, (size_t)buf_len - 1);
    buf[buf_len - 1] = '\0';
    return 0;
}

long long hk_cuda_device_total_mem(void) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
    return (long long)total_b;
}

long long hk_cuda_device_free_mem(void) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
    return (long long)free_b;
}

void* hk_cuda_malloc(size_t nbytes) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, nbytes) != cudaSuccess) return nullptr;
    return ptr;
}

int hk_cuda_upload(void* dev_ptr, const void* host_ptr, size_t nbytes) {
    cudaError_t err = cudaMemcpy(dev_ptr, host_ptr, nbytes, cudaMemcpyHostToDevice);
    return (int)err;
}

int hk_cuda_download(void* host_ptr, const void* dev_ptr, size_t nbytes) {
    cudaError_t err = cudaMemcpy(host_ptr, dev_ptr, nbytes, cudaMemcpyDeviceToHost);
    return (int)err;
}

void hk_cuda_free(void* dev_ptr) {
    if (dev_ptr) cudaFree(dev_ptr);
}

int hk_cuda_synchronize(void) {
    return (int)cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------
// F32 GEMV: y[r] = dot(W[r, :], x), W is [M, K] row-major.
// One block per output row; block-wide tree reduction in shared memory.
// ---------------------------------------------------------------------
__global__ void gemvF32Kernel(const float* __restrict__ W,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int K) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const float* row_ptr = W + (size_t)row * (size_t)K;

    float partial = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        partial += row_ptr[i] * x[i];
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

int hk_cuda_gemv_f32(const float* d_W, const float* d_x, float* d_y, int M, int K) {
    const int threads = 256;
    const size_t shmem = threads * sizeof(float);
    gemvF32Kernel<<<M, threads, shmem>>>(d_W, d_x, d_y, K);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Q8_0 GEMV: matches tensor_ops.zig's BlockQ8_0 layout exactly:
//   34 bytes/block = 2-byte f16 scale `d` + 32 int8 quantized values,
//   one block covers 32 elements of K.
// One block per output row; each thread handles a stride of 32-wide
// quant blocks, block-wide reduction at the end.
// ---------------------------------------------------------------------
__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

__global__ void gemvQ8_0Kernel(const uint8_t* __restrict__ W_bytes,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int blocks_per_row) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const size_t row_bytes_len = (size_t)blocks_per_row * 34u;
    const uint8_t* row_ptr = W_bytes + (size_t)row * row_bytes_len;

    float partial = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* blk = row_ptr + (size_t)b * 34u;
        uint16_t d_raw;
        std::memcpy(&d_raw, blk, 2);
        const float d = f16_to_f32(d_raw);
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* x_sub = x + b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            block_sum += (float)qs[i] * x_sub[i];
        }
        partial += block_sum * d;
    }

    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sdata[0];
}

int hk_cuda_gemv_q8_0(const uint8_t* d_W_bytes, const float* d_x, float* d_y, int M, int K) {
    const int blocks_per_row = K / 32;
    const int threads = 128;
    const size_t shmem = threads * sizeof(float);
    gemvQ8_0Kernel<<<M, threads, shmem>>>(d_W_bytes, d_x, d_y, blocks_per_row);
    return (int)cudaGetLastError();
}

} // extern "C"
