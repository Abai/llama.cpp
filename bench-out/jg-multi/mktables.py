#!/usr/bin/env python3
"""Assemble JG-style markdown tables (one per KV combo / ladder) from the
*.sql files produced by benchmark.sh, across all GPUs."""
import re, glob, collections, os, sys, datetime

# Usage: mktables.py [dir ...] - renders rungs from the given directories
# (e.g. "master-<mbase> <feature>"); with no args, the script's own directory
# (legacy flat layout).
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DIRS = sys.argv[1:] if len(sys.argv) > 1 else ['.']

FNAME = re.compile(r'^(?P<gpu>[A-Za-z0-9]+)-(?P<model>qwen|llama|gemma|deepseek)-'
                   r'(?P<kv>q80-?q80|q40-?q40|q80-?q40|f16)'
                   r'(?:-(?P<lad>ladpp|ladtg))?(?:-d(?P<d>\d+))?-'
                   r'(?P<side>master|branch|branch16|branch32|branch128|fix|fix32)'
                   r'(?:-ub(?P<ub>\d+))?\.sql$')

def parse_sql(path):
    s = open(path, errors='replace').read()
    m = re.search(r'INSERT INTO llama_bench \(([^)]*)\) VALUES \((.*)\);', s, re.S)
    if not m:
        return None
    return dict(zip(m.group(1).split(', '),
                    re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(2))))

KVLABEL = {'q80q80': 'q8_0/q8_0', 'q40q40': 'q4_0/q4_0', 'q80q40': 'q8_0/q4_0', 'f16': 'f16/f16'}
GPUNAMES = {'RTX3090': 'RTX 3090', 'RTX5070LaptopGPU': 'RTX 5070 Laptop', 'RTX5070Laptop': 'RTX 5070 Laptop', 'RTX5070': 'RTX 5070 Laptop'}
SIDE_ORDER = [('branch16', '16MiB'), ('branch32', '32MiB'), ('branch', '64MiB'), ('branch128', '128MiB'), ('fix32', 'fix 32MiB'), ('fix', 'fix 64MiB')]

data = collections.defaultdict(dict)  # (kv, lad, depth) -> (gpu, model, sortkey) -> {side: (ts, label, test)}
for f in sorted(f for d in DIRS for f in glob.glob(os.path.join(d, '*.sql'))):
    fm = FNAME.match(os.path.basename(f))
    if not fm:
        continue
    d = parse_sql(f)
    if not d:
        print(f'WARN: no data in {f}')
        continue
    g = fm.groupdict()
    g['kv'] = g['kv'].replace('-', '')
    depth = int(d['n_depth'])
    n_gen, n_prompt = int(d['n_gen']), int(d['n_prompt'])
    test = f'tg{n_gen}@d{depth}' if n_gen > 0 else f'pp{n_prompt}@d{depth}'
    if g['lad']:
        key = (g['kv'], g['lad'], -1)      # ladder groups collect all depths
        row = (g['gpu'], g['model'], depth)
    else:
        key = (g['kv'], '', depth)
        row = (g['gpu'], g['model'], int(g['ub']))
    reps = None
    try:
        mm = re.search(r' r(\d+)\s*$', open(f + '.bin').read())
        reps = mm.group(1) if mm else None
    except OSError:
        pass
    data[key].setdefault(row, {})[g['side']] = (float(d['avg_ts']), d['model_type'], test, d['n_ubatch'], reps, d.get('test_time', ''))

PAIR_MAX_GAP_H = 6          # master/branch measured further apart than this is not a valid pair
warnings = []
def parse_time(s):
    try:
        return datetime.datetime.fromisoformat(s.replace('Z', ''))
    except Exception:
        return None

kv_order = ['q80q80', 'q40q40', 'q80q40', 'f16']
def group_sort(k):
    kv, lad, depth = k
    return (kv_order.index(kv), lad != '', lad, depth)

for key in sorted(data, key=group_sort):
    kv, lad, depth = key
    rows = data[key]
    sides = [(s, lbl) for (s, lbl) in SIDE_ORDER
             if s == 'branch' or any(s in r for r in rows.values())]
    caps = len(sides) > 1
    title = {'ladpp': 'pp512 depth ladder', 'ladtg': 'tg128 depth ladder'}.get(lad, f'pp1024@d{depth}')
    print(f'\n### KV {KVLABEL[kv]}, {title}\n')
    ubcol = 'Depth' if lad else 'Microbatch size'
    # repetitions column only when the table mixes different -r values
    reps_seen = {v[4] for rr in rows.values() for v in rr.values() if v[4]}
    mixed = len(reps_seen) > 1
    rcol = ' reps |' if mixed else ''
    rsep = '-----:|' if mixed else ''
    def reps_cell(r):
        vals = [r[s][4] or '?' for s in ('master', 'branch') if s in r]
        return (' ' + '/'.join(vals) + ' |') if mixed else ''
    if caps:
        hdr = ' | '.join(f't/s PR {lbl}' for _, lbl in sides)
        spd = ' | '.join(f'Speedup {lbl}' for _, lbl in sides)
        print(f'| GPU | Model | {ubcol} | Test | t/s master | {hdr} | {spd} |' + rcol)
        print(f'|:----|:------|--------:|:-----|-----------:|' + '-----------:|'*(2*len(sides)) + rsep)
    else:
        print(f'| GPU | Model | {ubcol} | Test | t/s master | t/s PR | Speedup |' + rcol)
        print(f'|:----|:------|--------:|:-----|-----------:|-----------:|--------:|' + rsep)
    for (gpu, model, sk) in sorted(rows):
        r = rows[(gpu, model, sk)]
        gl = GPUNAMES.get(gpu, gpu)
        if 'master' not in r or 'branch' not in r:
            ncols = 2*len(sides) + 1 if caps else 3
            print(f'| {gl} | {model} | {sk} | ? |' + ' MISSING |'*ncols + (' - |' if mixed else ''))
            continue
        mts, mtype, test = r['master'][0], r['master'][1], r['master'][2]
        # provenance guard: a ratio is only meaningful if both sides were measured
        # in the same session with the same binaries (see benchmark.sh, srckey)
        for s_ in [x for x in r if x != 'master']:
            ta, tb = parse_time(r['master'][5]), parse_time(r[s_][5])
            if ta and tb and abs((tb - ta).total_seconds()) > PAIR_MAX_GAP_H*3600:
                warnings.append(f"{gl} {model} {sk} {kv} {s_}: master measured {ta:%Y-%m-%d %H:%M}, "
                                f"{s_} {tb:%Y-%m-%d %H:%M} - {abs((tb-ta).total_seconds())/3600:.1f} h apart")
        if caps:
            ts_cells, spd_cells = [], []
            for s, _l in sides:
                if s in r:
                    ts = r[s][0]
                    ts_cells.append(f'{ts:.2f}')
                    spd_cells.append(f'{ts/mts:.2f}')
                else:
                    ts_cells.append('-')
                    spd_cells.append('-')
            print(f'| {gl} | {mtype} | {sk} | {test} | {mts:.2f} | ' +
                  ' | '.join(ts_cells) + ' | ' + ' | '.join(spd_cells) + ' |' + reps_cell(r))
        else:
            bts = r['branch'][0]
            print(f'| {gl} | {mtype} | {sk} | {test} | {mts:.2f} | {bts:.2f} | {bts/mts:.2f} |' + reps_cell(r))

for w in warnings:
    print(f'WARN: pair measured across sessions: {w}')
