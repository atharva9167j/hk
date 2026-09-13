#ifndef HK_H
#define HK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef HK_BUILDING_DLL
    #define HK_API __declspec(dllexport)
  #else
    #define HK_API __declspec(dllimport)
  #endif
#else
  #define HK_API __attribute__((visibility("default")))
#endif

typedef enum {
    HK_STORAGE_F32 = 0x00,
    HK_STORAGE_F16 = 0x01,
    HK_STORAGE_BF16 = 0x02,
    HK_STORAGE_FP8_E4M3 = 0x03,
    HK_STORAGE_FP8_E5M2 = 0x04,
    HK_STORAGE_INT8 = 0x05,
    HK_STORAGE_INT32 = 0x06,
    HK_STORAGE_INT64 = 0x07,
    HK_STORAGE_UINT8 = 0x08,
    HK_STORAGE_BOOL = 0x09,
    HK_STORAGE_DQ4 = 0x10,
    HK_STORAGE_DQ8 = 0x11,
    HK_STORAGE_DQ6 = 0x12,
    HK_STORAGE_DQ12 = 0x13,
    HK_STORAGE_DQT = 0x14,
    HK_STORAGE_Q4_0 = 0x15,
    HK_STORAGE_Q8_0 = 0x16,
    HK_STORAGE_SPARSE_F16 = 0x20,
    HK_STORAGE_SPARSE_DQ8 = 0x21,
    HK_STORAGE_SPARSE_2_4 = 0x22,
    HK_STORAGE_SPARSE_DQ4_2_4 = 0x23,
    HK_STORAGE_NULL_REF = 0x30,
    HK_STORAGE_SHARED_REF = 0x31,
    HK_STORAGE_LORA_REF = 0x32,

    // Advanced K-Quants (256-element super-blocks)
    HK_STORAGE_Q2_K = 0x40,
    HK_STORAGE_Q3_K = 0x41,
    HK_STORAGE_Q4_K = 0x42,
    HK_STORAGE_Q5_K = 0x43,
    HK_STORAGE_Q6_K = 0x44,
    HK_STORAGE_Q8_K = 0x45,

    // Advanced I-Quants
    HK_STORAGE_IQ1_S = 0x50,
    HK_STORAGE_IQ1_M = 0x51,
    HK_STORAGE_IQ2_XXS = 0x52,
    HK_STORAGE_IQ2_XS = 0x53,
    HK_STORAGE_IQ3_XXS = 0x54,
    HK_STORAGE_IQ4_NL = 0x55,
    HK_STORAGE_IQ4_XS = 0x56,

    // Microscaling & Ternary Formats
    HK_STORAGE_TQ1_0 = 0x60,
    HK_STORAGE_TQ2_0 = 0x61,
    HK_STORAGE_MXFP4 = 0x62,
    HK_STORAGE_NVFP4 = 0x63
} hk_storage_type_t;

typedef enum {
    HK_TILE_ROW_MAJOR = 0x00,
    HK_TILE_COL_MAJOR = 0x01,
    HK_TILE_16X16 = 0x02,
    HK_TILE_16X8 = 0x03,
    HK_TILE_32X16 = 0x04,
    HK_TILE_BLOCK_SPARSE_2_4 = 0x05,
    HK_TILE_32X32 = 0x06,
    HK_TILE_64X64 = 0x07
} hk_tile_layout_t;

typedef enum {
    HK_SPARSITY_NONE = 0x00,
    HK_SPARSITY_BITMASK = 0x01,
    HK_SPARSITY_CSR = 0x02,
    HK_SPARSITY_STRUCTURED_2_4 = 0x03,
    HK_SPARSITY_PHYSICAL_PRUNED = 0x04,
    HK_SPARSITY_BSR = 0x05
} hk_sparsity_type_t;

typedef struct {
    const char* name;
    uint8_t storage_type;
    uint8_t tile_layout;
    uint8_t sparsity_type;
    uint8_t ndim;
    uint64_t shape[8];
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t residual_offset;
    uint64_t residual_size;
    uint64_t scale_offset;
    uint64_t scale_size;
    uint16_t block_size;
    float sparsity_ratio;
} hk_tensor_info_t;

typedef struct hk_reader_t hk_reader_t;

// Lifecycle & File Access
HK_API hk_reader_t* hk_open(const char* path);
HK_API void hk_close(hk_reader_t* reader);

// Tensor Access
HK_API uint64_t hk_get_tensor_count(const hk_reader_t* reader);
HK_API int hk_get_tensor_info(const hk_reader_t* reader, uint64_t index, hk_tensor_info_t* out_info);
HK_API const void* hk_get_tensor_data(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);
HK_API const void* hk_get_tensor_residual(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);
HK_API const void* hk_get_tensor_scales(const hk_reader_t* reader, uint64_t index, uint64_t* out_size);

// Dequantization & Precision Recovery
HK_API int hk_dequantize_f32(
    const hk_reader_t* reader,
    uint64_t index,
    int with_residual,
    float* out_buf,
    uint64_t count
);

// Metadata
HK_API const char* hk_get_metadata_string(const hk_reader_t* reader, const char* key);
HK_API int hk_get_metadata_int(const hk_reader_t* reader, const char* key, int64_t* out_val);
HK_API int hk_get_metadata_float(const hk_reader_t* reader, const char* key, double* out_val);
HK_API int hk_get_metadata_bool(const hk_reader_t* reader, const char* key, int* out_val);

// Sharding & Multi-File Container Support
#define HK_FLAG_IS_SHARDED 0x40

HK_API int hk_reader_is_sharded(const hk_reader_t* reader);
HK_API uint16_t hk_reader_get_split_index(const hk_reader_t* reader);
HK_API uint16_t hk_reader_get_split_count(const hk_reader_t* reader);

// In-Place Key-Value Metadata Patching
HK_API int hk_metadata_patch_in_place(const char* file_path, const char* key, const char* val);

// Low-level Quantization Helpers
HK_API float hk_quantize_block_nf4(const float* block, uint32_t count, uint8_t* packed_out, float* residual_out);
HK_API float hk_quantize_block_dq8(const float* block, uint32_t count, int8_t* out_i8, float* residual_out);
HK_API float hk_quantize_block_dqt(const float* block, uint32_t count, uint8_t* packed_out, float* residual_out);

// Sparsity & Tiling Helpers
HK_API int hk_pack_2_4(const float* dense_in, uint64_t count, uint8_t* out_bytes);
HK_API int hk_unpack_2_4(const uint8_t* payload, uint64_t payload_len, uint64_t count, float* out_buf);
HK_API int hk_tile_16x16_pack(const float* in_row_major, uint64_t M, uint64_t K, float* out_tiled);
HK_API int hk_tile_16x16_unpack(const float* in_tiled, uint64_t M, uint64_t K, float* out_row_major);

// Appendix Definitions
typedef enum {
    HK_APPENDIX_LORA_ADAPTER = 0x01,
    HK_APPENDIX_DELTA_PATCH = 0x02,
    HK_APPENDIX_NEW_LAYER = 0x03,
    HK_APPENDIX_CODE_EVAL = 0x04,
    HK_APPENDIX_KV_CACHE_SINK = 0x05,
    HK_APPENDIX_TOPOLOGY_HEAD = 0x06
} hk_appendix_type_t;

typedef struct {
    uint8_t entry_type;
    uint8_t flags;
    uint32_t generation;
    uint64_t timestamp;
    uint8_t parent_hash[32];
    float metric_loss;
    float metric_acc;
    float metric_pass;
    float metric_custom;
    const char* name;
    const char* target;
    const void* data;
    uint64_t data_size;
} hk_appendix_entry_t;

// Appendix Access & Mutation
HK_API uint64_t hk_appendix_get_count(const hk_reader_t* reader);
HK_API int hk_appendix_get_entry(const hk_reader_t* reader, uint64_t index, hk_appendix_entry_t* out_entry);
HK_API int hk_appendix_append(
    const char* file_path,
    uint8_t entry_type,
    uint8_t flags,
    const char* name,
    const char* target,
    uint32_t generation,
    const uint8_t parent_hash[32],
    float metric_loss,
    float metric_acc,
    float metric_pass,
    float metric_custom,
    const void* data,
    uint64_t data_size
);
HK_API int hk_appendix_rollback(const char* file_path, uint32_t target_generation);

// Native SIMD Tensor Operations & Inference Kernels
HK_API float hk_dot_product_f32(const float* a, const float* b, uint64_t count);
HK_API void hk_gemv_f32(
    const float* W,
    const float* x,
    const float* bias,
    float* y,
    uint64_t M,
    uint64_t K
);
HK_API void hk_gemm_f32(
    const float* A,
    const float* B,
    float* C,
    uint64_t M,
    uint64_t K,
    uint64_t N
);
HK_API void hk_fused_gemv_nf4(
    const uint8_t* packed_W,
    const float* scales,
    const float* x,
    const float* bias,
    float* y,
    uint64_t M,
    uint64_t K,
    uint32_t block_size
);
HK_API void hk_fused_gemv_dq8(
    const int8_t* W_i8,
    const float* scales,
    const float* x,
    const float* bias,
    float* y,
    uint64_t M,
    uint64_t K,
    uint32_t block_size
);

// Native Dynamic Architecture Growth (Net2Net)
HK_API int hk_net2wider(
    const float* w_in_old,
    const float* b_in_old,
    float* w_in_new,
    float* b_in_new,
    const float* w_out_old,
    float* w_out_new,
    uint64_t old_out,
    uint64_t new_out,
    uint64_t in_f,
    uint64_t out_f,
    float noise_std,
    uint64_t seed
);
HK_API void hk_net2deeper(float* weights, float* bias, uint64_t dim);

HK_API int hk_net2wider_swiglu(
    const float* w_gate_old,
    float* w_gate_new,
    const float* b_gate_old,
    float* b_gate_new,
    const float* w_up_old,
    float* w_up_new,
    const float* b_up_old,
    float* b_up_new,
    const float* w_down_old,
    float* w_down_new,
    const float* b_down_old,
    float* b_down_new,
    uint64_t old_inter,
    uint64_t new_inter,
    uint64_t in_features,
    uint64_t out_features,
    int zero_init,
    float noise_std,
    uint64_t seed
);

HK_API int hk_expand_vocab(
    const float* embed_old,
    float* embed_new,
    const float* lm_head_old,
    float* lm_head_new,
    uint64_t old_vocab,
    uint64_t new_vocab,
    uint64_t hidden_dim,
    uint64_t seed
);

HK_API void hk_plasticity_mask_rows(float* grad, uint64_t total_elements, uint64_t cutoff_rows, uint64_t cols);
HK_API void hk_plasticity_mask_cols(float* grad, uint64_t total_elements, uint64_t rows, uint64_t cutoff_cols, uint64_t cols);

// SIMD Vector Activations & Normalizations
HK_API int hk_forward_swiglu(
    const float* x,
    const float* w_gate,
    const float* b_gate,
    const float* w_up,
    const float* b_up,
    const float* w_down,
    const float* b_down,
    float* intermediate_buf,
    float* out,
    uint64_t in_features,
    uint64_t inter_features,
    uint64_t out_features
);
HK_API void hk_forward_rmsnorm(const float* x, const float* weight, float eps, float* out, uint64_t n);
HK_API void hk_forward_silu(const float* x, float* out, uint64_t n);

// Container Writer API
typedef struct hk_writer_t hk_writer_t;

HK_API hk_writer_t* hk_writer_create(uint64_t alignment);
HK_API void hk_writer_destroy(hk_writer_t* writer);
HK_API void hk_writer_set_sharding(hk_writer_t* writer, uint16_t split_index, uint16_t split_count);
HK_API int hk_writer_add_metadata_string(hk_writer_t* writer, const char* key, const char* val);
HK_API int hk_writer_add_metadata_int(hk_writer_t* writer, const char* key, int64_t val);
HK_API int hk_writer_add_metadata_float(hk_writer_t* writer, const char* key, double val);
HK_API int hk_writer_add_metadata_bool(hk_writer_t* writer, const char* key, int val);
HK_API int hk_writer_add_tensor(
    hk_writer_t* writer,
    const char* name,
    uint8_t storage_type,
    uint8_t tile_layout,
    uint8_t sparsity_type,
    uint8_t ndim,
    const uint64_t* shape,
    const uint8_t* data,
    uint64_t data_len,
    float sparsity_ratio
);
HK_API int hk_writer_write_to_file(hk_writer_t* writer, const char* path);

// Advanced Quantization & Microscaling C ABI
HK_API int hk_quantize_block_q4_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q4_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q8_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q8_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_q6_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_q2_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_iq4_nl(const uint8_t* packed_in, float scale, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_mxfp4(const uint8_t* packed_in, uint8_t scale_e8m0, uint32_t count, float* out_f32);
HK_API int hk_dequantize_block_nvfp4(const uint8_t* packed_in, uint8_t scale_fp8, uint32_t count, float* out_f32);

// GGUF Quantization & Dequantization C ABI
HK_API int hk_quantize_block_q4_0(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q4_0(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q8_0(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q8_0(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q5_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q5_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q3_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_dequantize_block_q3_k(const void* in_block, uint32_t count, float* out_f32);
HK_API int hk_quantize_block_q6_k(const float* weights, uint32_t count, void* out_block);
HK_API int hk_quantize_block_q2_k(const float* weights, uint32_t count, void* out_block);

// Packed-Weight SIMD GEMV Kernels
HK_API int hk_gemv_q8_0(const void* w_packed, const float* x, const float* bias, float* out, uint64_t m, uint64_t k);
HK_API int hk_gemv_q4_0(const void* w_packed, const float* x, const float* bias, float* out, uint64_t m, uint64_t k);
HK_API int hk_gemv_q4_k(const void* w_packed, const float* x, const float* bias, float* out, uint64_t m, uint64_t k);

// Mathematical Tensor Transformations
HK_API int hk_rope_permute_hf_to_gguf(const float* in_w, float* out_w, uint64_t n_heads, uint64_t head_dim, uint64_t batch_size);
HK_API int hk_rope_unpermute_gguf_to_hf(const float* in_w, float* out_w, uint64_t n_heads, uint64_t head_dim, uint64_t batch_size);
HK_API int hk_layernorm_offset(const float* in_w, float* out_w, uint64_t count, float offset);

// GGUF Transcoder C ABI
HK_API int hk_convert_gguf(const char* in_gguf_path, const char* out_hk_path);
HK_API int hk_export_gguf(const char* in_hk_path, const char* out_gguf_path);

#ifdef __cplusplus
}
#endif

#endif // HK_H
