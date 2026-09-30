SASS gate for feature/cuda-fattn-convert-buffer-cap (54b94d245, tree
89799b487758ce7f250ce68ce4c6431d5fc92cc0) vs master 36b101543, sm_86, 2026-09-25.

Result: 1068 main FA kernels instruction-identical, 82 NO_DEVICE_CODE stubs identical
after masking source-location immediates, 0 DIFF, 0 MISSING. GATE: PASS.

Scope. The branch compiles two MMA variants per kernel, write_partial false and true;
master compiles one. The gate pairs by kernel NAME rather than by order, mapping a branch
name with three trailing template bools <use_logit_softcap, V_is_K_view, write_partial>
onto master's two-bool name by dropping the last, and compares only write_partial == false
against master. vec and tile kernel names are unchanged. So this gate proves exactly the
claim the PR makes: every launch that does NOT chunk - f16/f32 KV, anything fattn-vec
handles, and quantized KV whose conversion fits under the cap - runs master's instructions.
It says nothing about the chunked path, which is new code by construction.

Builds: cap-src/build-cap (branch) and cap-src/build-master (master 36b101543), image
llamacpp-bench-dev:12.8.1-gcc14 (76081e77f985). Reuses parse_sass / parse_resources / stub masking
from scripts/sass-diff-fattn-helper.py. Run inside that image; cuobjdump must be on PATH.

An earlier run on 2026-09-17, from the then-separate capv2 worktree, produced an identical
result (1068 identical, 0 DIFF); that scratch directory is no longer kept. That worktree was
folded into this branch on 2026-09-22 as a commit-level change: new parent, same tree, so
the earlier gate already covered the shipped tree. This run reproduces it from the current
build directories, under a name that reflects the branch rather than the retired "v2" one.
