#!/usr/bin/env python3
"""Render VRAM (compute-buffer) tables from the *-vram-*.log files produced
by benchmark.sh --vram. Usage: mkvramtables.py <dir> [dir ...]"""
import re, glob, collections, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DIRS = sys.argv[1:] if len(sys.argv) > 1 else ['.']

FNAME = re.compile(r'^(?P<gpu>[A-Za-z0-9]+)-(?P<model>qwen|llama|gemma|deepseek)-vram-'
                   r'(?P<kv>q80-?q80|q40-?q40|q80-?q40|f16)-c(?P<ctx>\d+)-'
                   r'(?P<side>master|branch|branch32)\.log$')

def parse_log(path):
    s = open(path, errors='replace').read()
    # multiple lines can appear (e.g. an initial 0.00 KV line): take the max
    mm = [float(x) for x in re.findall(r'CUDA0 compute buffer size\s*=\s*([0-9.]+) MiB', s)]
    kvv = [float(x) for x in re.findall(r'CUDA0 KV buffer size\s*=\s*([0-9.]+) MiB', s)]
    ok = 'listening on' in s
    oom = bool(re.search(r'failed to allocate|out of memory|cudaMalloc failed', s))
    timeout = '# vram-probe: TIMEOUT' in s
    return {'compute': max(mm) if mm else None,
            'kv': max(kvv) if kvv else None,
            'ok': ok, 'oom': oom, 'timeout': timeout}

KVLABEL = {'q80q80': 'q8_0/q8_0', 'q40q40': 'q4_0/q4_0', 'q80q40': 'q8_0/q4_0', 'f16': 'f16/f16'}
GPUNAMES = {'RTX3090': 'RTX 3090', 'RTX5070LaptopGPU': 'RTX 5070 Laptop'}
SIDES = ['master', 'branch', 'branch32']

data = collections.defaultdict(dict)  # kv -> (gpu, model, ctx) -> side -> parsed
for f in sorted(x for d in DIRS for x in glob.glob(os.path.join(d, '*-vram-*.log'))):
    fm = FNAME.match(os.path.basename(f))
    if not fm:
        continue
    g = fm.groupdict()
    kv = g['kv'].replace('-', '')
    data[kv].setdefault((g['gpu'], g['model'], int(g['ctx'])), {})[g['side']] = parse_log(f)

def cell(p):
    if p is None: return '-'
    if p['timeout']: return 'timeout'
    if p['oom'] and p['compute'] is None: return 'OOM'
    if p['compute'] is None: return '?'
    tag = ' (OOM later)' if p['oom'] and not p['ok'] else ''
    return f"{p['compute']:.2f}{tag}"

for kv in ['q80q80', 'q40q40', 'q80q40', 'f16']:
    if kv not in data: continue
    rows = data[kv]
    has32 = any('branch32' in r for r in rows.values())
    print(f'\n### VRAM: CUDA0 compute buffer (MiB), KV {KVLABEL[kv]}, -ub 512\n')
    hdr = '| GPU | Model | n_ctx | master | PR (64 MiB cap) |'
    sep = '|:----|:------|------:|-------:|--------------------:|'
    if has32: hdr += ' PR (32 MiB cap) |'; sep += '---------:|'
    hdr += ' saved MiB | KV buffer MiB |'; sep += '----------:|--------------:|'
    print(hdr); print(sep)
    for (gpu, model, ctx) in sorted(rows):
        r = rows[(gpu, model, ctx)]
        gl = GPUNAMES.get(gpu, gpu)
        m, b = r.get('master'), r.get('branch')
        saved = '-'
        if m and b and m.get('compute') is not None and b.get('compute') is not None:
            saved = f"{m['compute'] - b['compute']:.2f}"
        kvbuf = '-'
        for p in (b, m):
            if p and p.get('kv') is not None: kvbuf = f"{p['kv']:.2f}"; break
        line = f'| {gl} | {model} | {ctx} | {cell(m)} | {cell(b)} |'
        if has32: line += f" {cell(r.get('branch32'))} |"
        line += f' {saved} | {kvbuf} |'
        print(line)
print()
print('OOM = context failed to load (allocation failure); a numeric value with')
print('(OOM later) means the reserve pass reported the size before a later')
print('allocation failed. Measured via llama-server --no-warmup (n_outputs_max=1).')
