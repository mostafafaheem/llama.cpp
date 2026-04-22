# Why `--context-shift` Cannot Be Supported in Stateful Mode

## What is context shift?

When the KV cache is full (all `n_ctx` slots occupied), context shift frees space for continued generation:

1. The first `n_keep` tokens (typically just the BOS token) are preserved.
2. The next `n_discard = (n_ctx - n_keep) / 2` tokens are **logically removed** — their KV cache slots are marked empty but the physical data is left in place.
3. The remaining tokens have their positions shifted back by `n_discard` and their K vectors receive an in-place RoPE correction.

Critically, `seq_rm` does **not** compact memory. After context shift, the physical KV cache layout has holes:

```
slot:   0       1 .. 127       128 .. 254     255
data:  [BOS]   [stale/freed]  [shifted]      [empty]
pos:    0       -1 (empty)     1 .. 127       -1
```

The freed slots are masked out with `-inf` during attention, making the stale data invisible. New tokens are later placed into these freed slots by `find_slot`.

## Why it breaks stateful mode

In stateful mode, the OpenVINO model manages the KV cache internally as an opaque state tensor with shape `[batch, kv_len, heads, head_dim]`. The KV entries are stored **contiguously** — position `i` in the state tensor corresponds to the `i`-th token in sequence order.

The mask slicing in `add_sliced_mask` (translate_session.cpp) relies on this contiguity:

```cpp
// stateful mask slicing: take mask[:, :, :, 0 : last_pos+1]
auto last_inp_pos_inc = last_inp_pos + 1;
mask_sliced = Slice(mask, 0, last_inp_pos_inc, 1, axis=-1);
```

This assumes:
- State tensor rows `[0, kv_len)` are all valid and in position order.
- The mask only needs columns `[0, last_pos+1)` because every column before `last_pos+1` corresponds to a real KV entry.

After context shift, this assumption fails:
- The ggml KV cache has holes at slots `[n_keep, n_keep + n_discard)`.
- Valid entries are non-contiguous: slots `[0, n_keep)` and `[n_keep + n_discard, n_past)`.
- The attention mask from llama.cpp has `-inf` columns for the freed slots, but the stateful model's sliced mask discards those columns entirely.

Even if we compact the KV data before writing it back to the state (as the current code does via `get_kv_shift_info` + memcpy), the **mask is still wrong**: llama.cpp generates the mask for the full `n_ctx`-wide non-contiguous layout, but the stateful model expects a compact `kv_len`-wide mask where every column is valid.

## Summary

| Aspect | Non-stateful (explicit KV) | Stateful (opaque KV) |
|--------|---------------------------|---------------------|
| KV layout after shift | Holes (freed slots with stale data) | N/A — state is always contiguous |
| Mask | Full `n_ctx` width, `-inf` for holes | Sliced to `kv_len`, assumes all valid |
| Compaction | Not needed (mask handles holes) | Would be required, but mask mismatch remains |

To properly support context shift in stateful mode, we would need to either:
1. Regenerate the mask to match the compacted state (not feasible — the mask comes from llama.cpp's KV cell metadata), or
2. Bypass llama.cpp's context shift entirely and implement a stateful-native eviction strategy.

Neither is practical, so `--context-shift` should be disabled when using stateful OpenVINO inference.
