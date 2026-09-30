# L2 cap ladder — RTX 3090 (GA102, 6 MiB L2): negative result

Counterpart to l2-gb206/. Tests whether an L2-sized cap reproduces the
GB206 residency speedup on Ampere. It does not: llama 8B q8_0/q8_0
pp1024@d32768, caps 2/3/4/6/8/12/16 MiB (r=3, containerized, branch
binary 330690fea) are STRICTLY monotonic — smaller cap always worse,
down to 0.41x master at 2 MiB; no inflection near the 6 MiB L2.
cudaDeviceProp.l2CacheSize verified = 6291456.

Mechanism: an L2-sized chunk on GA102 is only ~1.5k KV rows -> 22+
chunks at d32768; per-chunk overhead dominates, and a 6 MB scratch
cannot stay resident against competing Q/weight/mask traffic anyway.
The GB206 win requires a usefully-large chunk (>= ~8 MiB) fitting L2 —
a precondition absent below ~16 MiB of L2.

Raw: cap<N>M-ub{64,512}.sql. Reference points from the main matrix:
master 1608.56/2235.65, branch64 1531.26/2183.23 (ub64/ub512).
