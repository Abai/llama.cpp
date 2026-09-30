CUDA: cap FA convert buffer for quantized KV

The fattn-mma-f16 and fattn-tile kernels ingest f16 KV data layer by layer. For
non-f16 (quantized) KV types this requires a conversion buffer, pre-allocated
per #23907. This size of this buffer is equal to one layer of f16 KV and scales
with ctx length. For quantized KV types, this negates large part of VRAM savings
obtained by quantization in the first place.

Implement serial split-KV - Cap the buffer to a fixed size (64 MiB, env override
GGML_CUDA_FATTN_CONVERT_BYTES for testing only), process KV in chunks re-using
the buffer. For each processed KV chunk fattn kernel emits an unnormalized
partial and (max_row_val, row_sum), these get folded into running accumulators
by flash_attn_combine_results kernel found in fattn-common.

KV chunking is enabled based on two decisions: Firstly, in addition to picking
the best fattn kernel, ggml_cuda_flash_attn_ext_get_alloc_size() also sets the
kernel_supports_chunking flag to true, currently enabled for fattn-mma-f16 and
fattn-tile kernels. Secondly, ggml_cuda_fattn_type_chunk_enabled()
disables/enables chunking depending on other criteria such as KV type.
Currently, it returns true for quantized KV types only.

For launches with KV types f16 and f32, launches with fattn-vec kernel and
launches with FA convert buffer size below the GGML_CUDA_FATTN_CONVERT_BYTES
cap, the outputs are bit-identical to master.

Before this commit, the scratch buffer costs one layer of f16 KV. For
Qwen3.6-27B at ctx 131072, for example, this negates 13.3% of the VRAM savings
at q8_0 and 8.7% at q4_0 KV quantization. After this commit the cost is bounded
at 64 MiB max, regardless of context length.

The f16 conversion of quantized KV chunk, similar to master, picks between
to_fp16_cuda and to_fp16_nc_cuda conversion kernels depending on the KV chunk's
allocation contiguity.

Assisted-by: Claude Fable 5

