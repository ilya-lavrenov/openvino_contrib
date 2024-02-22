#include "openvino/runtime/tensor.hpp"

void paged_attention_v1_cpu(
    ov::Tensor out, ov::Tensor query, ov::Tensor key_cache,
    ov::Tensor value_cache, int num_kv_heads, float scale,
    ov::Tensor block_tables, ov::Tensor context_lens, ov::Tensor alibi_slopes,
    int block_size, int max_context_len);

void reshape_and_cache_cpu(ov::Tensor key, ov::Tensor value,
                           ov::Tensor key_cache, ov::Tensor value_cache,
                           ov::Tensor slot_mapping);
