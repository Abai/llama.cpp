SASS gate, feature tree of 74c907e3d (the shipped tip, including the Blackwell-only
config retune) vs master 36b101543, sm_86, 2026-09-25.
Objects: the 21 fattn-mma-f16-instance-ncols1_*-ncols2_*.cu.o (f16 instances) from the campaign builds
build-cap (feature) and build-master, image llamacpp-bench-dev:12.8.1-gcc14 (76081e77f985).
Pairing by kernel order inside each object via scripts/sass-diff-fattn-helper.py diff; result 21/21 identical.

Supersedes an earlier gate run on 2026-09-18 against the pre-retune tree (c6886f107);
that artifact is no longer kept.
The retune adds a Blackwell-only override guarded by cc >= GGML_CUDA_CC_BLACKWELL on the host
and __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL on the device, so it compiles out entirely on sm_86
and the f16 instances remain byte-identical to master.
