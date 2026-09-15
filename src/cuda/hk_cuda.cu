// HK CUDA backend: High-performance device-side kernels for full GPU inference
// and dynamic CPU/GPU offloading.
//
// Supports:
// - Optimized GEMV (F32, Q8_0, Q4_0) with warp shuffles (__shfl_down_sync)
//   and vectorized 128-bit memory transactions (float4 / int4).
// - Full inference pipeline: RMSNorm, QK-Norm, RoPE, KV cache update,
//   Grouped-Query Attention (GQA) with online softmax, SwiGLU, and residual add.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

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
// Warp & Block Level Reductions
// ---------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Max 32 warps per block (1024 threads)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Half-precision (f16) to float conversion
__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
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

// ---------------------------------------------------------------------
// Optimized GEMV F32: y[r] = dot(W[r, :], x)
// Vectorized float4 memory loads + warp shuffle reduction
// ---------------------------------------------------------------------
__global__ void gemvF32KernelOpt(const float* __restrict__ W,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int K) {
    const int row = blockIdx.x;
    const float* row_ptr = W + (size_t)row * (size_t)K;

    float sum = 0.0f;
    const int K4 = K / 4;
    const float4* W4 = reinterpret_cast<const float4*>(row_ptr);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (int i = threadIdx.x; i < K4; i += blockDim.x) {
        float4 w_val = W4[i];
        float4 x_val = x4[i];
        sum += w_val.x * x_val.x + w_val.y * x_val.y + w_val.z * x_val.z + w_val.w * x_val.w;
    }

    for (int i = K4 * 4 + threadIdx.x; i < K; i += blockDim.x) {
        sum += row_ptr[i] * x[i];
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        y[row] = sum;
    }
}

int hk_cuda_gemv_f32(const float* d_W, const float* d_x, float* d_y, int M, int K) {
    const int threads = 256;
    gemvF32KernelOpt<<<M, threads>>>(d_W, d_x, d_y, K);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Optimized GEMV Q8_0: 34 bytes per block (2-byte f16 scale + 32 int8)
// ---------------------------------------------------------------------
__global__ void gemvQ8_0KernelOpt(const uint8_t* __restrict__ W_bytes,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int blocks_per_row) {
    const int row = blockIdx.x;
    const size_t row_bytes_len = (size_t)blocks_per_row * 34u;
    const uint8_t* row_ptr = W_bytes + (size_t)row * row_bytes_len;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* blk = row_ptr + (size_t)b * 34u;
        uint16_t d_raw;
        std::memcpy(&d_raw, blk, 2);
        const float d = f16_to_f32(d_raw);
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* x_sub = x + b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            block_sum += (float)qs[i + 0] * x_sub[i + 0]
                       + (float)qs[i + 1] * x_sub[i + 1]
                       + (float)qs[i + 2] * x_sub[i + 2]
                       + (float)qs[i + 3] * x_sub[i + 3];
        }
        sum += block_sum * d;
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        y[row] = sum;
    }
}

int hk_cuda_gemv_q8_0(const uint8_t* d_W_bytes, const float* d_x, float* d_y, int M, int K) {
    const int blocks_per_row = K / 32;
    const int threads = 256;
    gemvQ8_0KernelOpt<<<M, threads>>>(d_W_bytes, d_x, d_y, blocks_per_row);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Optimized GEMV Q4_0: 18 bytes per block (2-byte f16 scale + 16-byte nibbles)
// ---------------------------------------------------------------------
__global__ void gemvQ4_0KernelOpt(const uint8_t* __restrict__ W_bytes,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int blocks_per_row) {
    const int row = blockIdx.x;
    const size_t row_bytes_len = (size_t)blocks_per_row * 18u;
    const uint8_t* row_ptr = W_bytes + (size_t)row * row_bytes_len;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* blk = row_ptr + (size_t)b * 18u;
        uint16_t d_raw;
        std::memcpy(&d_raw, blk, 2);
        const float d = f16_to_f32(d_raw);
        const uint8_t* qs = blk + 2;
        const float* x_sub = x + b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = qs[i];
            int8_t q0 = (int8_t)(byte & 0x0F) - 8;
            int8_t q1 = (int8_t)((byte >> 4) & 0x0F) - 8;
            block_sum += (float)q0 * x_sub[i] + (float)q1 * x_sub[i + 16];
        }
        sum += block_sum * d;
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        y[row] = sum;
    }
}

int hk_cuda_gemv_q4_0(const uint8_t* d_W_bytes, const float* d_x, float* d_y, int M, int K) {
    const int blocks_per_row = K / 32;
    const int threads = 256;
    gemvQ4_0KernelOpt<<<M, threads>>>(d_W_bytes, d_x, d_y, blocks_per_row);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Full-dimension RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
// ---------------------------------------------------------------------
__global__ void rmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ out,
                              int dim,
                              float eps) {
    __shared__ float s_inv_rms;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = blockReduceSum(sum_sq);
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

int hk_cuda_rmsnorm(const float* d_x, const float* d_weight, float* d_out, int dim, float eps) {
    const int threads = 256;
    rmsNormKernel<<<1, threads>>>(d_x, d_weight, d_out, dim, eps);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Per-Head RMSNorm (QK-Norm for Qwen3 / Gemma 2)
// Normalizes each head independently before RoPE
// ---------------------------------------------------------------------
__global__ void headRmsNormKernel(float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  int n_heads,
                                  int head_dim,
                                  int weight_len,
                                  float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    float* head_ptr = x + h * head_dim;
    const float* w_ptr = (weight_len == head_dim) ? weight : (weight + h * head_dim);

    __shared__ float s_inv_rms;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = head_ptr[i];
        sum_sq += v * v;
    }
    sum_sq = blockReduceSum(sum_sq);
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head_ptr[i] = head_ptr[i] * inv_rms * w_ptr[i];
    }
}

int hk_cuda_head_rmsnorm(float* d_x, const float* d_weight, int n_heads, int head_dim, int weight_len, float eps) {
    const int threads = 128;
    headRmsNormKernel<<<n_heads, threads>>>(d_x, d_weight, n_heads, head_dim, weight_len, eps);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// RoPE: Rotary Position Embedding for Q and K
// ---------------------------------------------------------------------
__global__ void ropeKernel(float* __restrict__ q,
                           float* __restrict__ k,
                           int pos,
                           int n_heads,
                           int n_kv_heads,
                           int head_dim,
                           float rope_theta) {
    const int half_dim = head_dim / 2;
    const int pair_idx = threadIdx.x;
    const int h = blockIdx.x;

    if (pair_idx >= half_dim) return;

    float freq = 1.0f / powf(rope_theta, (2.0f * (float)pair_idx) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    if (h < n_heads) {
        float* q_head = q + h * head_dim;
        float v0 = q_head[2 * pair_idx];
        float v1 = q_head[2 * pair_idx + 1];
        q_head[2 * pair_idx]     = v0 * cos_val - v1 * sin_val;
        q_head[2 * pair_idx + 1] = v0 * sin_val + v1 * cos_val;
    }

    if (h < n_kv_heads) {
        float* k_head = k + h * head_dim;
        float v0 = k_head[2 * pair_idx];
        float v1 = k_head[2 * pair_idx + 1];
        k_head[2 * pair_idx]     = v0 * cos_val - v1 * sin_val;
        k_head[2 * pair_idx + 1] = v0 * sin_val + v1 * cos_val;
    }
}

int hk_cuda_rope(float* d_q, float* d_k, int pos, int n_heads, int n_kv_heads, int head_dim, float rope_theta) {
    const int half_dim = head_dim / 2;
    const int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;
    ropeKernel<<<max_heads, half_dim>>>(d_q, d_k, pos, n_heads, n_kv_heads, head_dim, rope_theta);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// KV-Cache Update on Device
// ---------------------------------------------------------------------
__global__ void kvCacheUpdateKernel(float* __restrict__ key_cache,
                                    float* __restrict__ val_cache,
                                    const float* __restrict__ k,
                                    const float* __restrict__ v,
                                    int layer,
                                    int pos,
                                    int max_seq_len,
                                    int kv_dim) {
    const int safe_pos = (pos < max_seq_len) ? pos : (max_seq_len - 1);
    const size_t offset = ((size_t)layer * (size_t)max_seq_len + (size_t)safe_pos) * (size_t)kv_dim;

    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        key_cache[offset + i] = k[i];
        val_cache[offset + i] = v[i];
    }
}

int hk_cuda_kv_cache_update(float* d_key_cache, float* d_val_cache, const float* d_k, const float* d_v,
                            int layer, int pos, int max_seq_len, int kv_dim) {
    const int threads = 256;
    kvCacheUpdateKernel<<<1, threads>>>(d_key_cache, d_val_cache, d_k, d_v, layer, pos, max_seq_len, kv_dim);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Grouped-Query Attention (GQA) with Online Softmax & V-Accumulation
// ---------------------------------------------------------------------
__global__ void gqaAttentionKernel(const float* __restrict__ q,
                                   const float* __restrict__ key_cache,
                                   const float* __restrict__ val_cache,
                                   float* __restrict__ out,
                                   int layer,
                                   int pos,
                                   int n_heads,
                                   int n_kv_heads,
                                   int head_dim,
                                   int max_seq_len,
                                   int kv_dim) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    extern __shared__ float s_scores[];
    const int max_t = (pos + 1 < max_seq_len) ? (pos + 1) : max_seq_len;
    const int n_rep = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    const int kv_head_idx = (n_rep > 0) ? (h / n_rep) : 0;
    const float scale = 1.0f / sqrtf((float)head_dim);

    const float* q_head = q + h * head_dim;
    const size_t layer_kv_base = (size_t)layer * (size_t)max_seq_len * (size_t)kv_dim;

    // 1. Compute dot product attention scores
    for (int t = threadIdx.x; t < max_t; t += blockDim.x) {
        const float* k_head = key_cache + layer_kv_base + (size_t)t * (size_t)kv_dim + (size_t)kv_head_idx * (size_t)head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_head[d];
        }
        s_scores[t] = dot * scale;
    }
    __syncthreads();

    // 2. Numerical stable Softmax
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < max_t; t += blockDim.x) {
        if (s_scores[t] > local_max) local_max = s_scores[t];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    static __shared__ float s_warp_max[32];
    if ((threadIdx.x % 32) == 0) s_warp_max[threadIdx.x / 32] = local_max;
    __syncthreads();

    static __shared__ float s_max_val;
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_max[threadIdx.x] : -1e30f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (threadIdx.x == 0) s_max_val = val;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float max_val = s_max_val;
    for (int t = threadIdx.x; t < max_t; t += blockDim.x) {
        float e = expf(s_scores[t] - max_val);
        s_scores[t] = e;
        local_sum += e;
    }
    local_sum = blockReduceSum(local_sum);
    static __shared__ float s_inv_sum;
    if (threadIdx.x == 0) {
        s_inv_sum = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;
    }
    __syncthreads();

    float inv_sum = s_inv_sum;
    for (int t = threadIdx.x; t < max_t; t += blockDim.x) {
        s_scores[t] *= inv_sum;
    }
    __syncthreads();

    // 3. Weighted sum over V
    float* out_head = out + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < max_t; t++) {
            const float* v_head = val_cache + layer_kv_base + (size_t)t * (size_t)kv_dim + (size_t)kv_head_idx * (size_t)head_dim;
            acc += s_scores[t] * v_head[d];
        }
        out_head[d] = acc;
    }
}

int hk_cuda_gqa_attention(const float* d_q, const float* d_key_cache, const float* d_val_cache,
                          float* d_out, int layer, int pos, int n_heads, int n_kv_heads,
                          int head_dim, int max_seq_len, int kv_dim) {
    const int threads = 128;
    const int max_t = (pos + 1 < max_seq_len) ? (pos + 1) : max_seq_len;
    const size_t shmem = max_t * sizeof(float);
    gqaAttentionKernel<<<n_heads, threads, shmem>>>(d_q, d_key_cache, d_val_cache, d_out, layer, pos,
                                                    n_heads, n_kv_heads, head_dim, max_seq_len, kv_dim);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// SwiGLU Activation: gate = SiLU(gate) * up
// ---------------------------------------------------------------------
__global__ void swigluKernel(float* __restrict__ gate,
                             const float* __restrict__ up,
                             int hidden_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        gate[idx] = silu_g * up[idx];
    }
}

int hk_cuda_swiglu(float* d_gate, const float* d_up, int hidden_dim) {
    const int threads = 256;
    const int blocks = (hidden_dim + threads - 1) / threads;
    swigluKernel<<<blocks, threads>>>(d_gate, d_up, hidden_dim);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------
// Vectorized Residual Addition: x = x + residual
// ---------------------------------------------------------------------
__global__ void addResidualKernel(float* __restrict__ x,
                                  const float* __restrict__ residual,
                                  int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        x[idx] += residual[idx];
    }
}

int hk_cuda_add_residual(float* d_x, const float* d_residual, int dim) {
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    addResidualKernel<<<blocks, threads>>>(d_x, d_residual, dim);
    return (int)cudaGetLastError();
}

} // extern "C"
