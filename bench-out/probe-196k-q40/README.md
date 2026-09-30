# Max-context probes at the current pair (2026-09-11)

Re-measurement of the issue's max-context motivation bullet after the
packetD vram-ladder (old master pair, UD-Q4_K_XL weights) was pruned.
Method: does llama-server start (-ngl 99 -fa on -ub 512 --no-warmup,
KV q8_0/q8_0) at a given -c. RTX 3090 24 GB, driver 580.173.02, image
llamacpp-bench-dev:12.8.1-gcc14. master = build-master at 36b101543
(binary fingerprint-identical to the jg-multi VRAM campaign); fused =
built 2026-09-11 from the implementation-commit tree (ff8cc2867).
probe.sh is the runner; logs are filtered to the allocation-relevant
lines.

Qwen3.6-27B-Q4_0 (14.9 GiB weights):

| side | starts | fails |
|---|---|---|
| master | 229376 (compute 1200.28 MiB) | 237568 |
| fused | 253952 (compute 370.28 MiB) | 262144 |

Qwen3.6-27B-UD-Q4_K_XL (16.1 GiB weights):

| side | starts | fails |
|---|---|---|
| master | 180224 (compute 960.28 MiB) | 184320 |
| fused | 196608 (compute 314.28 MiB) | 212992 |

The XL rows reproduce the pruned packetD ladder digit-identically
(fused 314.28 MiB at 196608; master's 1040.28 MiB reservation fails
there), confirming the allocator is deterministic across the rebase.
The issue's "~229k to ~254k" bullet uses the Q4_0 rows; the historical
"180k to ~196k" figures were specific to the heavier XL quant.
