#!/usr/bin/env python3
"""SASS gate for feature/cuda-fattn-convert-buffer-cap, paired by kernel NAME (not order).

The branch instance objects hold two MMA variants per kernel (write_partial false/true), master holds one,
so the stock order-based gate misaligns. Here a a branch name with three trailing template bools
<..., use_logit_softcap, V_is_K_view, write_partial> is mapped to master's two-bool name by dropping
the last bool; only write_partial == false is compared against master. vec/tile names are unchanged.
Reuses parse_sass / parse_resources / stub masking from scripts/sass-diff-fattn-helper.py.
Run inside the toolchain image (cuobjdump on PATH).
"""
import sys, os, re, glob, importlib.util, difflib
BASE = '/home/abai/Projects/abai/online/workspace/llama.cpp'
spec = importlib.util.spec_from_file_location('h', f'{BASE}/scripts/sass-diff-fattn-helper.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)

INST = 'ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances'
# v2 was folded into feature/cuda-fattn-convert-buffer-cap on 2026-09-22 and the
# capv2-src worktree was removed, so the feature side is now cap-src/build-cap.
# C (the old v1 build, kept for "v1=ident/DIFF" annotations) no longer exists;
# leaving it pointed at a missing dir just yields "v1=?" on every line.
A = f'{BASE}/cap-src/build-cap/{INST}'        # the shipped branch (54b94d245)
B = f'{BASE}/cap-src/build-master/{INST}'     # master 36b101543, same image, sm_86
C = f'{BASE}/cap-src/build-capv1-gone/{INST}' # v1 build, removed; annotations degrade to v1=?

MMA3 = re.compile(r'^(_Z18flash_attn_ext_f16I(?:Li\d+E){4}(?:Lb[01]E){2})Lb([01])E(E.*)$')
def keyed(items, strip_wp):
    out, wp1 = {}, 0
    for name, body in items:
        m = MMA3.match(name) if strip_wp else None
        if m:
            if m.group(2) == '1': wp1 += 1; continue
            name = m.group(1) + m.group(3)
        out[name] = body
    return out, wp1
def stubmask(body): return [h.FLT_RE.sub('FIMM', h.IMM_RE.sub('0xIMM', ln)) for ln in body]
def is_main(n): return any(t in n for t in ('flash_attn_ext_f16I', 'flash_attn_tile', 'flash_attn_ext_vec'))

rc = 0; tot = {'ident':0,'stub':0,'diff':0,'missing':0}; helpers_diff = []
objs = sorted(os.path.basename(p) for p in glob.glob(f'{A}/fattn-*-instance-*.cu.o'))
for obj in objs:
    pa, pb, pc = f'{A}/{obj}', f'{B}/{obj}', f'{C}/{obj}'
    if not os.path.exists(pb): print(f'{obj}: not in master build (new instance file), skipped'); continue
    sa, wp1 = keyed(h.parse_sass(pa), True); sb, _ = keyed(h.parse_sass(pb), False)
    ra, _ = keyed(h.parse_resources(pa), True); rb, _ = keyed(h.parse_resources(pb), False)
    sc = keyed(h.parse_sass(pc), False)[0] if os.path.exists(pc) else {}
    fam = obj.split('-instance-')[0]
    line = []
    for name, bb in sb.items():
        if not is_main(name):
            if name in sa and sa[name] != bb: helpers_diff.append((obj, name))
            continue
        if name not in sa: tot['missing'] += 1; rc = 1; print(f'  MISSING in v2: {obj} {name[:80]}'); continue
        ba = sa[name]
        v1 = 'prev=ident' if sc.get(name) == bb else ('prev=DIFF' if name in sc else 'prev=?')
        if ba == bb:
            if ra.get(name) != rb.get(name): rc = 1; print(f'  RES-DIFF {obj} {name[:70]}\n    v2: {ra.get(name)}\n    m : {rb.get(name)}')
            tot['ident'] += 1; continue
        stub = not any(h.WORK_RE.search(ln) for ln in ba + bb)
        if stub and stubmask(ba) == stubmask(bb): tot['stub'] += 1; continue
        tot['diff'] += 1; rc = 1
        print(f'  SASS-DIFF {obj} ({v1}) {len(ba)} vs {len(bb)} instr\n    {name}')
        for ln in list(difflib.unified_diff(ba, bb, 'v2', 'master', lineterm='', n=1))[:24]: print('    ' + ln)
    print(f'{obj:58s} main kernels vs master: {sum(1 for n in sb if is_main(n) and n in sa and sa[n]==sb[n]):3d} identical'
          + (f', {wp1} write_partial=true variants (not compared)' if wp1 else ''))
print()
print(f'TOTAL main FA kernels: identical={tot["ident"]}  stub-identical(masked __LINE__)={tot["stub"]}  DIFF={tot["diff"]}  MISSING={tot["missing"]}')
print(f'helper kernels differing from master (fixup/combine, expected): {len(helpers_diff)}')
for obj, n in helpers_diff[:6]: print(f'    {obj}: {n[:90]}')
print('GATE:', 'PASS' if rc == 0 else 'FAIL')
sys.exit(rc)
