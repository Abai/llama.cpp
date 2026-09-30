# L2 residency investigation — RTX 5070 Laptop (GB206, 32 MiB L2)

Artifacts from the 2026-09-04 laptop investigation of the branch32
speedup anomaly (branch32 1.2-1.38x vs master on llama@d32768).

- prof32.csv / prof64.csv — ncu, 48 FA-kernel instances @d32768,
  --cache-control none; key metric lts__t_sector_op_read_hit_rate:
  ~87.8% (cap32) vs ~31.6% (cap64).
- nsys-{cap32,cap64,master}.csv — full-run kernel totals, llama q8_0/q8_0
  pp1024@d32768: FA 9.69 s / 13.48 s / 13.99 s; combine+fixup overhead
  0.60 s vs master 0.06 s; matmuls -12% at cap32 (cache/power headroom).

Findings:
chunk scratch <= L2 makes the convert->FA path L2-resident; cap sweep
peaks at 24 MiB (1.41x master, agent-reported, files not retained);
cap >= 128 MiB (unchunked) = exact master parity, proving zero overhead
on the unchunked path. f16 ub512@d16384 "0.96" was confirmed noise
(source + cuobjdump register parity + interleaved remeasure 0.96/1.00/1.01).
