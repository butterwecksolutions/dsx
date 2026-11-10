# Engram row prefetch in prefill (2026-10-26, 2x RTX 3090)

Engram lookup rows depend only on the token ids, so prefill no longer gathers them inline at layers 1 and 14 while
both GPUs wait. At the start of each chunk a helper thread hashes the chunk's n-grams and gathers both layers' rows
(layer 1's first, same 32-thread distinct-row reader as before); the layer loop takes them from a channel.
`DSX_ENGRAM_PREFETCH=0` restores the inline gather. The arithmetic is unchanged: final logits are byte-identical.

Idea from Project Maya's V4.1 prefill notes (Engram reads had capped their prefill at 250 tok/s).

## Results

`prefill-bench`, T16 v1 GGUF, 8K tokens in one 8192 chunk, ctx 262144, fixed placement, the 10-25 prefill profile
(`DSX_T16_WARP_HAD`, `DSX_T16_SPECIALIZED_DQ`, `DSX_PREFILL_REUSE_BUFFERS`, `DSX_T16_FUSED_POST_GU`). Each process
starts after dropping the page cache: round 0 reads Engram rows from NVMe (cold), rounds 1-2 from the page cache
(warm; CUDA first-use cost is also in round 0). Order base, pf, pf, base; tok/s.

| Prompt | Distinct rows / layer | Base cold | Prefetch cold | Base warm | Prefetch warm |
|---|---:|---:|---:|---:|---:|
| `prompt8000.ids` (repetitive fixture, 1,586 distinct trigrams) | 38K | 917.1, 918.5 | 953.3, 954.6 (+3.9%) | 1000.0-1002.8 | 1015.4-1016.6 (+1.5%) |
| `prompt8000_code.ids` (dsx server source + docs, 5,753 distinct trigrams) | 125K | 793.1, 791.3 | 866.2, 866.3 (+9.4%) | 917.8-920.0 | 936.3-938.3 (+2.0%) |

Inline gather cost per Engram layer (`DSX_PROF`, base): fixture 205 ms cold / 78 ms warm, code prompt 543 / 128 ms,
plus 2 ms hashing and 21 ms upload. With prefetch the layer loop waited 0 ms for rows in every profiled round
(`pf-prof-cold.log`), so the remaining 2 x 21 ms upload is the only Engram host time left on the critical path.

The old FINDINGS figure of ~4.4 s of Engram gather per 8K prefill predates the deduplicated reads and the IQ4_NL
table; today the stage cost 0.2-1.1 s before this change.

Files: `pfbench.sh` (runner), `*-prof*.log` (DSX_PROF), `base-*/pf-*` and `code-*` logs (timing), the code prompt ids.
