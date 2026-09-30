#!/usr/bin/env python3
"""Render perplexity / KL-divergence tables from the summary files written by
benchmark.sh --ppl / --kld. Usage: mkppltables.py <dir> [dir ...]"""
import re, glob, os, sys, collections
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DIRS = sys.argv[1:] if len(sys.argv) > 1 else ['.']
FN = re.compile(r'^(?P<gpu>[A-Za-z0-9]+)-(?P<model>qwen|llama|gemma|deepseek)-(?P<kind>ppl|kld)-'
                r'(?P<kv>f16|q80|q40)-c(?P<ctx>\d+)-n(?P<n>\d+)-(?P<side>master|branch)\.txt$')
GPUNAMES = {'RTX3090': 'RTX 3090', 'RTX5070LaptopGPU': 'RTX 5070 Laptop', 'RTX5070Laptop': 'RTX 5070 Laptop', 'RTX5070': 'RTX 5070 Laptop'}
MODELS = {'llama': 'llama 8B Q4_0', 'gemma': 'gemma 2B Q4_0', 'qwen': 'qwen35 27B Q4_0', 'deepseek': 'deepseek2 16B Q4_0'}
KV = {'f16': 'f16/f16', 'q80': 'q8_0/q8_0', 'q40': 'q4_0/q4_0'}
NUM = r'([-+]?\d+(?:\.\d+)?)'
def grab(text, pat):
    m = re.search(pat, text); return m.groups() if m else None
data = collections.defaultdict(dict)  # (kind, gpu, model, kv, ctx, n) -> side -> dict
for f in sorted(x for d in DIRS for x in glob.glob(os.path.join(d, '*.txt'))):
    m = FN.match(os.path.basename(f))
    if not m: continue
    t = open(f, errors='replace').read()
    r = {}
    g = grab(t, r'Final estimate: PPL = ' + NUM + r' \+/- ' + NUM)
    if g: r['ppl'], r['ppl_err'] = g
    g = grab(t, r'Mean\s+KLD:\s*' + NUM + r'\s*\S+\s*' + NUM);         r['kld'], r['kld_err'] = g if g else ('-', '-')
    g = grab(t, r'Median\s+KLD:\s*' + NUM);                             r['kld_med'] = g[0] if g else '-'
    g = grab(t, r'99\.0%\s+KLD:\s*' + NUM);                             r['kld_99'] = g[0] if g else '-'
    g = grab(t, r'Maximum KLD:\s*' + NUM);                              r['kld_max'] = g[0] if g else '-'
    g = grab(t, r'Same top p:\s*' + NUM + r'\s*\S+\s*' + NUM);          r['top'], r['top_err'] = g if g else ('-', '-')
    g = grab(t, r'Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*' + NUM + r'\s*\S+\s*' + NUM); r['lnr'], r['lnr_err'] = g if g else ('-', '-')
    data[(m['kind'], m['gpu'], m['model'], m['kv'], int(m['ctx']), int(m['n']))][m['side']] = r
ppl = {k: v for k, v in data.items() if k[0] == 'ppl'}
if ppl:
    for ctx, n in sorted({(k[4], k[5]) for k in ppl}):
        print(f'\n### Perplexity: wikitext-2 test, n_ctx={ctx}, {n} chunks ({ctx*n} tokens)\n')
        print('| GPU | Model | KV | PPL master | PPL PR | PR - master |')
        print('|:----|:------|:---|-----------:|-------:|------------:|')
        for k in sorted(ppl):
            if (k[4], k[5]) != (ctx, n): continue
            s = ppl[k]; a, b = s.get('master'), s.get('branch')
            fa = f"{a['ppl']} +/- {a['ppl_err']}" if a and 'ppl' in a else 'MISSING'
            fb = f"{b['ppl']} +/- {b['ppl_err']}" if b and 'ppl' in b else 'MISSING'
            d = f"{float(b['ppl']) - float(a['ppl']):+.4f}" if a and b and 'ppl' in a and 'ppl' in b else '-'
            print(f"| {GPUNAMES.get(k[1], k[1])} | {MODELS.get(k[2], k[2])} | {KV[k[3]]} | {fa} | {fb} | {d} |")
kld = {k: v for k, v in data.items() if k[0] == 'kld'}
if kld:
    for ctx, n in sorted({(k[4], k[5]) for k in kld}):
        print(f'\n### KL divergence vs the master f16-KV run: wikitext-2 test, n_ctx={ctx}, {n} chunks ({ctx*n} tokens)\n')
        print('| GPU | Model | KV | side | PPL | mean KLD | median KLD | 99% KLD | max KLD | same top-1 % | mean ln(PPL/PPL_base) |')
        print('|:----|:------|:---|:-----|----:|---------:|-----------:|--------:|--------:|-------------:|----------------------:|')
        for k in sorted(kld):
            if (k[4], k[5]) != (ctx, n): continue
            for side in ('master', 'branch'):
                r = kld[k].get(side)
                if not r: continue
                label = 'PR' if side == 'branch' else 'master'
                print(f"| {GPUNAMES.get(k[1], k[1])} | {MODELS.get(k[2], k[2])} | {KV[k[3]]} | {label} | {r.get('ppl','-')} | "
                      f"{r['kld']} +/- {r['kld_err']} | {r['kld_med']} | {r['kld_99']} | {r['kld_max']} | {r['top']} +/- {r['top_err']} | {r['lnr']} +/- {r['lnr_err']} |")
print()
