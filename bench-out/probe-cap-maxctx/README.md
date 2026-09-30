# Max-context probe, conversion-cap branch (2026-09-29)

Does `llama-server` start at a given `-c`? Success = "listening on", failure =
an allocation error. `-ngl 99 -fa on -ub 512 --no-warmup`, KV q8_0/q8_0,
RTX 3090 24 GB, driver 580.173.02, image `llamacpp-bench-dev:12.8.1-gcc14`.
master = `build-master` at 36b101543; cap = `cap-src/build-cap` at 54b94d245.
`probe.sh` is the runner; it mirrors `benchmark.sh run_vram`. Allocation is
deterministic, so one probe per configuration is sufficient.

Qwen3.6-27B-Q4_0 (14.9 GiB weights):

| side | starts | fails |
|---|---|---|
| master | 229376 (compute 1200.28 MiB) | 237568 (compute 1240.28 MiB) |
| cap | 253952 (compute 392.28 MiB) | 262144 (compute 400.28 MiB) |

The cap branch reaches 253952 where master stops at 229376, +24576 tokens
(+10.7%). At the context where master first fails, 237568, master's compute
buffer needs 1240.28 MiB against cap's 376.28 MiB - a difference of 864 MiB,
which is the unbounded f16 conversion buffer that this PR caps at 64 MiB.

The master boundary reproduces the 2026-09-11 measurement in
bench-out/probe-196k-q40/README.md (same 229376 / 237568 verdicts, same
1200.28 MiB compute buffer), so the two probes are directly comparable. The
fused-dequant branch reaches the same 253952 / 262144 boundary by removing the
conversion buffer entirely rather than bounding it; at 253952 its compute
buffer is 370.28 MiB against cap's 392.28 MiB, the difference being the 64 MiB
chunk scratch that cap still allocates. The 8192-token probe granularity is
coarser than that gap, so both land on the same boundary.
