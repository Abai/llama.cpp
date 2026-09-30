# Laptop aborting-cell recovery - hand-back (2026-09-25 14:30 UTC)

All 16 cells that aborted with Xid 8 in the 09-22/23 campaign are now measured on the laptop (RTX 5070 Laptop, sm_120, 8 GB):
7 fused + 9 cap, 41 rungs, every side stamped, 0 FAILED, 0 MISSING for the laptop, no RTX 5070 pair warning from the render.
CUDA graphs ON, --reps default (r3), benchmark.sh md5 ab47ed3a90ea21ff004bb89ac7edcfdc on both trees, untouched.
Nothing committed, pushed, or sent to the rig from the laptop. Merge is by hand with bench-out/merge-laptop.sh (see below).

## How the data was produced
- 2026-09-24 21:19Z - 09-25 07:28Z: cell-granular retry loop (bench-out/repair-loop.sh): each pass deleted every cell with a missing
  side or a >120 min pair gap (repair-laptop.sh --go), then ran --small on both trees. 12 cells recovered in passes 1-8.
- 07:28Z-08:32Z paused (user). 08:32Z resumed with the attempt caps lifted (user). 08:50Z-11:09Z a supervisor (bench-out/repair-skip.sh)
  killed 90 rungs whose cell already had a FAILED side in that pass (cell lost for the pass anyway); killed rungs are indistinguishable
  from real aborts on disk (.err only) and were all deleted whole with their cell before the next pass. No kept data was touched.
- 11:09Z: user switched to RUNG granularity - keep every side that passes, re-measure only aborted rungs, no --go between passes
  (bench-out/repair-loop-rung.sh). The last 4 cells closed in passes 27, 28, 36, 38; loop ended by itself 13:56:55Z.
- Running time 15 h 33 min (38 passes); 16 h 38 min wall including the pause. Xid 8 signature unchanged throughout
  (ggml-cuda.cu:107 CUDA error in graph_compute); recorded throughput is unaffected, see bench-out/diag-xid8/FINDINGS.md.

## Result per cell (raw values from the .sql files; rendered tables use 2 decimals)
| tree | cell | done in pass | master test_time | t/s master | branch (t/s, ratio) | branch32 (t/s, ratio) | max master-side gap |
|---|---|---|---|---|---|---|---|
| fused | llama-q80-q80-ub16 | 1 | 2026-09-24T21:27:31Z | 345.73 | 693.04, 2.005 | - | 1.2 min |
| fused | llama-q40-q40-ub8 | 1 | 2026-09-24T21:32:32Z | 189.13 | 400.61, 2.118 | - | 2.2 min |
| fused | llama-q80-q80-ub8 | 2 | 2026-09-24T22:53:54Z | 179.07 | 360.22, 2.012 | - | 2.3 min |
| fused | llama-q80-q40-ub4 | 2 | 2026-09-24T23:02:23Z | 98.56 | 230.55, 2.339 | - | 4.0 min |
| fused | llama-q80-q80-ub4 | 3 | 2026-09-25T00:03:58Z | 94.81 | 212.31, 2.239 | - | 4.2 min |
| fused | llama-q40-q40-ub4 | 4 | 2026-09-25T01:16:31Z | 100.87 | 249.30, 2.472 | - | 4.0 min |
| fused | gemma-q80q80-d131072-ub4 | 5 | 2026-09-25T02:19:05Z | 193.03 | 469.29, 2.431 | - | 6.7 min |
| cap | llama-q80-q80-ub64 | 1 | 2026-09-24T22:03:49Z | 709.32 | 737.38, 1.040 | 911.95, 1.286 | 1.1 min |
| cap | llama-q40-q40-ub64 | 1 | 2026-09-24T22:19:31Z | 733.81 | 759.00, 1.034 | 947.55, 1.291 | 1.1 min |
| cap | llama-q80-q80-ub8 | 3 | 2026-09-25T00:27:50Z | 178.98 | 184.57, 1.031 | 203.36, 1.136 | 4.6 min |
| cap | llama-q80-q40-ub8 | 4 | 2026-09-25T02:02:37Z | 185.43 | 191.50, 1.033 | 212.65, 1.147 | 4.5 min |
| cap | llama-q40-q40-ub8 | 8 | 2026-09-25T04:57:28Z | 189.17 | 194.94, 1.031 | 217.23, 1.148 | 4.4 min |
| cap | llama-q40-q40-ub4 | 27 | 2026-09-25T12:06:17Z | 100.92 | 105.11, 1.042 | 120.38, 1.193 | 7.9 min |
| cap | llama-q80-q80-ub4 | 28 | 2026-09-25T12:28:37Z | 94.82 | 98.94, 1.043 | 111.75, 1.179 | 50.2 min |
| cap | gemma-q80q80-d131072-ub4 | 36 | 2026-09-25T12:57:10Z | 193.02 | 199.89, 1.036 | 223.56, 1.158 | 46.1 min |
| cap | llama-q80-q40-ub4 | 38 | 2026-09-25T13:52:54Z | 98.57 | 102.59, 1.041 | 117.25, 1.190 | 154.4 min |

Within-rung stddev of all 41 rungs is at most 0.35% (40 of 41 under 0.31%).

## Timing validity
- Render rule (mktables.py PAIR_MAX_GAP_H = 6 h, master vs each side): ALL 222 master-side pairs on the laptop pass; largest is 154 min.
- Repair-script rule (repair-laptop.py GAP_MIN = 120 min): one pair fails, cap llama-q80-q40-ub4 master vs branch = 154 min
  (vs branch32 = 94 min). USER DECISION: left as is, the render accepts it. Therefore: NEVER run repair-laptop.sh --go on this data
  again, it would delete that cell whole. Its dry run still lists it:
      cap: 1 cells to repair (92 total)
        would delete 3 sides  gap-branch=154m        llama-q80-q40-ub4
- The 12 cell-granular cells are all <= 7.9 min; the rung-granular ones are 7.9 / 50.2 / 46.1 / 154.4 min.

## Artifacts (exactly the files this session changed: 125, all *RTX5070LaptopGPU*, verified by mtime)
main   bench-out/jg-multi/master-36b101543/            7 rungs  x (.sql .sql.bin .err) = 21
main   bench-out/jg-multi/cuda-fattn-mma-fused-dequant/ 7 rungs  x 3 = 21  + TABLES-RTX5070LaptopGPU-sm120-8GB.md (rendered 11:07:13Z)
cap    cap-src/bench-out/jg-multi/master-36b101543/     9 rungs  x 3 = 27
cap    cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/ 18 rungs x 3 = 54  + TABLES-RTX5070LaptopGPU-sm120-8GB.md (rendered 13:56:55Z)
Sidecar stamps are the current source keys: master 'src:dc7bf2f682cbb5f1 r3' (92+92), fused 'src:1a5421efbb3fffcc r3' (92),
cap 'src:afbec7bce3717a3d r3' (176). File counts per dir after the run: 92 / 92 / 92 / 176 RTX5070 .sql, 0 stray, 0 orphan .err.
Full md5 list (125 lines, md5sum -c format, repo-relative paths): bench-out/jg-multi/HANDBACK-cells-2026-09-25-MANIFEST.md
(md5 2196ce444bf235fb52b12554a890dcb2). The 41 .sql:
5cf8879be896b407afabf6ceed0b8d6a  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q80-master-ub16.sql
12c3285f1edf837c40143e7574ccb2c1  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-llama-q80-q80-branch-ub16.sql
3997165d12ae78f5f65571930c5a315c  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q40-q40-master-ub8.sql
c4cb8a51db5320c398414026664a3878  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-llama-q40-q40-branch-ub8.sql
70be9276c4be0053eb8d02b87bff189d  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q80-master-ub8.sql
a24c614dc7e5c506355fe2e5dec72e4c  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-llama-q80-q80-branch-ub8.sql
557fb65f8a6c974e42a574b49a9fc241  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q40-master-ub4.sql
1803a96e2233a74e7268c72f85f35b0e  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-llama-q80-q40-branch-ub4.sql
bda73b9e7c71879454c391d7e6a0c5c2  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q80-master-ub4.sql
42b6685ca4db57b44bb4e64e87db41fe  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-llama-q80-q80-branch-ub4.sql
51ff82d37b2d21b6ee204e4355f748ff  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q40-q40-master-ub4.sql
32fd877ae82c0512d80e67fb0a328dc5  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-llama-q40-q40-branch-ub4.sql
bdf9bd5e5d1af953f0691ca67522031b  bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-gemma-q80q80-d131072-master-ub4.sql
1fb00d7e099a2c7f4116b8bedb036e65  bench-out/jg-multi/cuda-fattn-mma-fused-dequant/RTX5070LaptopGPU-gemma-q80q80-d131072-branch-ub4.sql
b31f3a0d2d761d9ea2bccf61826fd1fe  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q80-master-ub64.sql
267bef18719e13e7ebe25a19b6890714  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q80-branch-ub64.sql
7766e517f6ae433ed5fe6020d46eea55  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q80-branch32-ub64.sql
be587f42d1d2b45fe8cbdd7b4aab6bc8  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q40-q40-master-ub64.sql
e6b2dc2ddaf96dd97301cee72569cf2e  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q40-q40-branch-ub64.sql
b95d02867dd7baefa60b2c0b310c646b  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q40-q40-branch32-ub64.sql
02953c9ca761695b67e98180d6dc1be6  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q80-master-ub8.sql
f02059db5e271d0ad0b03e11e06cd719  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q80-branch-ub8.sql
0d894cd8ac8530bc980548e5c8ff23b2  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q80-branch32-ub8.sql
a2b053fd34deade3809d6c58f0283bfc  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q40-master-ub8.sql
ddb49e665dada39fa40f98814dcc8848  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q40-branch-ub8.sql
b945c37d769672dbb6a5088f95371544  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q40-branch32-ub8.sql
c5fa946f2caeda26af179dfcbf1d5319  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q40-q40-master-ub8.sql
aa79688dfa83ab1db9a88460ae95f150  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q40-q40-branch-ub8.sql
948ac7d4db5d8f398e29141195904a3b  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q40-q40-branch32-ub8.sql
9af3ccb6562740d72d5af62ed5aeb9fa  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q40-q40-master-ub4.sql
860fd945768c5207d055a08fb27e4b3b  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q40-q40-branch-ub4.sql
44ab3c496b8a992ba31774a510a1cfb7  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q40-q40-branch32-ub4.sql
2a434f7307d06dfd354be3aa19daa999  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q80-master-ub4.sql
1d4ff49a91f9ecf58f7e533b0c340479  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q80-branch-ub4.sql
fe9c543f8908baf273fff1f3dadec848  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q80-branch32-ub4.sql
da973a48901305bc1e76a8de2fb499cd  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-gemma-q80q80-d131072-master-ub4.sql
3f3c1eba53370205d3a9da86685e300e  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-gemma-q80q80-d131072-branch-ub4.sql
4e899d221957c2f2e46108b96e749557  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-gemma-q80q80-d131072-branch32-ub4.sql
6497d763e21ba59c68bcc41bd4f36950  cap-src/bench-out/jg-multi/master-36b101543/RTX5070LaptopGPU-llama-q80-q40-master-ub4.sql
6bbe973ffd4f55805cfc019e8a30049d  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q40-branch-ub4.sql
b1d555ed2847c47dbc604918752701b0  cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/RTX5070LaptopGPU-llama-q80-q40-branch32-ub4.sql

## Merge (on the rig HOST, which has rsync + ssh to the laptop)
    sh bench-out/merge-laptop.sh          # dry run, itemised
    sh bench-out/merge-laptop.sh --go
Expected in the itemised list: the 125 files above plus HANDBACK-cells-2026-09-25.md and the MANIFEST. Any further *RTX5070LaptopGPU*
files listed are byte-identical re-transfers from earlier merges (rsync -a transfers on mtime difference, source wins); nothing named
RTX3090-*, benchmark.sh or mktables.py may appear. No --delete.

## Post-merge verification (on the rig, repo root, BEFORE re-rendering)
    md5sum -c bench-out/jg-multi/HANDBACK-cells-2026-09-25-MANIFEST.md | grep -v ': OK' || echo "all 177 OK"
    sh bench-out/repair-laptop.sh          # dry run only: expect fused 0 cells, cap 1 cell (q80-q40-ub4, gap-branch=154m)
Then re-render both trees on the rig (the transferred TABLES-RTX5070LaptopGPU-sm120-8GB.md files were rendered on the laptop, whose
main tree still holds stale RTX 3090 copies: the laptop-rendered fused table shows 1 MISSING RTX 3090 row and its render printed
76 RTX 3090 pair warnings; none of that concerns the laptop rows). After the rig's render: expect 0 MISSING and 0 WARN for RTX 5070.
Rendered laptop rows to expect: fused llama d32768 ub4/8/16 speedups 2.00-2.47, gemma d131072 ub4 2.43; cap branch 1.03-1.04,
branch32 1.14-1.30 for all nine cells (consistent with the neighbouring rungs already on the rig).

## Not transferred (laptop only, all gitignored except the scripts and the state file)
bench-out/run-repair-loop.log (full loop log with PAUSED/RESUMED/decision markers), bench-out/run-repair-skip.log (the 90 kills),
bench-out/jg-multi/run-repair-passN-fused.log and cap-src/bench-out/jg-multi/run-repair-passN-cap.log (N = 1..38),
bench-out/repair-loop.sh, repair-loop-rung.sh, repair-skip.sh, repair-watch.sh, repair-loop.state.
Known but harmless: bench-out/repair-laptop.py defines an XID exclusion set (line 39) that the script never consults; it was not needed.

## Addendum 2026-09-25 17:51-17:56 UTC: the 8 stale cap PPL/KLD branch rungs re-measured
Requested by the user. Run: `PPL_CTX=16384 sh cap-src/bench-out/jg-multi/benchmark.sh --kld --models $M` then the same with
`--ppl` (PPL_CTX=16384 is the laptop protocol: without it --ppl targets NEW llama c32768 rungs on both sides instead of the
stale c16384 ones; gemma clamps to c8192 either way). Logs: cap-src/bench-out/jg-multi/run-kld-cap.log, run-ppl-cap.log.
Result: 8/8 done, 0 FAILED, 0 WARN. The 8 branch rungs now carry the current cap key 'src:afbec7bce3717a3d' (were
'src:20d4bcab44496e96'); the 8 masters ('src:dc7bf2f682cbb5f1', 2026-09-22) were skipped and are untouched.
PPL-RTX5070LaptopGPU-sm120-8GB.md re-rendered 17:56:24Z: 0 MISSING, 0 WARN; every laptop "PR - master" is +0.0000 and every
new final estimate equals the stale one to all printed digits (only provenance changed):
  gemma c8192 n9  f16 438.2826 / q8_0 438.1430 / q4_0 441.9798   llama c16384 n9  f16 5.7306 / q8_0 5.7303 / q4_0 5.8087
  llama KLD c4096 n4  q8_0 mean 0.000573 max 0.027627 / q4_0 mean 0.012769 max 0.790731
Artifacts: 26 files under cap-src/bench-out/jg-multi/cuda-fattn-convert-buffer-cap/, all *RTX5070LaptopGPU* (8 .txt, 8 .txt.bin,
9 .log incl. RTX5070LaptopGPU-llama-kld-base.log, 1 PPL table); appended to the MANIFEST (now 151 lines). Nothing else changed.

## Addendum 2026-09-25 18:16-18:22 UTC: the 8 stale FUSED PPL/KLD branch rungs re-measured
Same procedure on the main tree: `PPL_CTX=16384 sh bench-out/jg-multi/benchmark.sh --kld --models $M` then `--ppl`.
Logs: bench-out/jg-multi/run-kld-fused.log, run-ppl-fused.log. 8/8 done, 0 FAILED, 0 WARN. The 8 branch rungs now carry the
current fused key 'src:1a5421efbb3fffcc' (were 'src:f0cf08c5ad027845', 2026-09-21); the 8 masters skipped, untouched.
PPL-RTX5070LaptopGPU-sm120-8GB.md (fused) re-rendered 18:21:44Z: 0 MISSING, 0 WARN. Laptop rows now:
  gemma c8192 n9  f16 438.2826 (+0.0000) / q8_0 438.6653 (+0.5223) / q4_0 433.6014 (-8.3784)
  llama c16384 n9 f16 5.7306 (+0.0000) / q8_0 5.7307 (+0.0004) / q4_0 5.8021 (-0.0066)
  llama KLD c4096 n4  q8_0 mean 0.000602 max 0.019079 / q4_0 mean 0.012487 max 0.803367
7 of 8 final estimates equal the stale values to all digits. The ONE change: gemma q8_0/q8_0 PPL 438.9909 -> 438.6653
(delta vs master +0.8479 -> +0.5223). Expected: the retune folded into 74c907e3d changed exactly the gemma q8/q8 fused kernel
(see bench-out/retune-115w/FINDINGS-phaseB.md), so this rung's stale stamp covered a real source difference; f16, q4/q4 and all
llama rungs are unaffected by that retune and reproduce bit-for-bit.
Artifacts: 26 files under bench-out/jg-multi/cuda-fattn-mma-fused-dequant/, all *RTX5070LaptopGPU* (8 .txt, 8 .txt.bin, 9 .log incl.
RTX5070LaptopGPU-llama-kld-base.log, 1 PPL table); appended to the MANIFEST (now 177 lines). Nothing else changed.
