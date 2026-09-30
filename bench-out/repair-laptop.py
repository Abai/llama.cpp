#!/usr/bin/env python3
"""Repair RTX 5070 Laptop cells whose master/branch sides were measured far apart.

Two causes, both from the 2026-09-22/23 laptop campaign:

  * the partition bug (fixed 2026-09-23): rung_partition classified master by the
    CELL rule but feature sides per SIDE, so a cell could split across the --small
    and --identity runs. gemma at d32768 is the trigger - 33 MiB fits under the
    64 MiB cap (branch identical to master) but chunks at 32 MiB (branch32
    differs), so master went to one phase and branch to the other ~5 h later.
    These cleared the 6 h guard by ~45 min and so were never reported, but they
    sit 260x the 1.2 min median gap.
  * retry passes: a rung the Xid 8 watchdog killed was re-measured hours after
    its partner.

Deletes ALL sides of an affected cell (master + branch + branch32) so the next
run re-measures the triple intact - deleting one side only would leave the new
measurement paired against an old partner, which is the defect itself.

The Xid 8 cells are EXCLUDED: their master cannot be measured on this hardware
(RC watchdog, not OOM), so their branch data is the only data there will ever be
and must be kept. Their stale masters were deleted on the rig instead, so those
cells render without a ratio.

Dry run by default; pass --go to delete.

  sh bench-out/repair-laptop.sh            # list
  sh bench-out/repair-laptop.sh --go       # delete
"""
import sys, os, glob, sqlite3, datetime, collections

REPO = os.environ.get('REPO', '/home/abai/Projects/abai/online/workspace/llama.cpp')
# minutes. Chosen from the observed distribution, which has an EMPTY band:
# 219 pairs under 60 min, zero between 60 and 120, 42 above. Sub-60 gaps are
# legitimate - a cell is measured master -> branch -> branch32 in sequence and
# model loading dominates in-kernel time, so a triple can legitimately span
# ~45 min. Do not lower this: it would delete good data.
GAP_MIN = 120.0
# Cells deliberately accepted as-is: the check below would flag them, but a human
# has decided to keep them. Each entry needs a reason, because every one of these is
# a documented weakness in the dataset rather than a clean result.
#   llama-q80-q40-ub4 : recovered 2026-09-25 by RUNG-granular retry, so its master and
#     branch sides are 154 min apart. mktables' PAIR_MAX_GAP_H = 6 h accepts it; this
#     script's GAP_MIN = 120 min does not. Without this entry, --go would delete the
#     cell that took 38 passes to obtain.
SKIP = {'llama-q80-q40-ub4'}
TREES = [('fused', f'{REPO}/bench-out/jg-multi/', 'cuda-fattn-mma-fused-dequant'),
         ('cap',   f'{REPO}/cap-src/bench-out/jg-multi/', 'cuda-fattn-convert-buffer-cap')]
GO = '--go' in sys.argv


def test_time(path):
    con = sqlite3.connect(':memory:')
    con.executescript(open(path, errors='replace').read())
    r = con.execute('select test_time from llama_bench').fetchone()
    return datetime.datetime.strptime(r[0], '%Y-%m-%dT%H:%M:%SZ') if r else None


def side_of(name):
    for s in ('-branch32', '-branch', '-master'):
        if s in name:
            return s
    return None


# A rung whose .err exists but whose .sql does not FAILED: run_lb removes the
# .sql and keeps the .err. These are invisible to the gap check below (a cell
# with no master is skipped), which is exactly how 13 failed rungs hid behind
# the rig's stale copies. Report them first.
def failed_rungs(d):
    return sorted(os.path.basename(e)[:-4] for e in glob.glob(f'{d}/RTX5070*.err')
                  if not os.path.exists(e[:-4] + '.sql'))


total = 0
for label, base, fdir in TREES:
    for d in (base + fdir, base + 'master-36b101543'):
        f = failed_rungs(d)
        if f:
            print(f'  {label}: {len(f)} FAILED rungs (.err, no .sql) in {os.path.basename(d)}')
            for n in f:
                print(f'      {n}')
    fdirp, mdirp = base + fdir, base + 'master-36b101543'
    if not os.path.isdir(fdirp):
        print(f'  {label}: {fdirp} missing, skipped')
        continue

    # Group every rung by cell, keeping the full path stem of each side. Index BOTH
    # .sql and .err: a side that aborted has only an .err, and if we indexed .sql
    # alone that side would be invisible - which is exactly the blind spot that let
    # 13 failed rungs hide behind the rig's stale copies.
    cells = collections.defaultdict(dict)
    for d in (fdirp, mdirp):
        for ext in ('.sql', '.err'):
            for p in glob.glob(f'{d}/RTX5070*{ext}'):
                n = os.path.basename(p)[:-len(ext)]
                s = side_of(n)
                if s:
                    cells[n.replace(s, '')][s] = p[:-len(ext)]

    bad = []
    for cell, sides in sorted(cells.items()):
        short = cell.replace('RTX5070LaptopGPU-', '')
        if short in SKIP:
            continue
        # A cell is bad if ANY side is missing its .sql (the rung aborted) OR any
        # pair straddles GAP_MIN. Both need the WHOLE cell re-measured: measuring
        # one side alone is what manufactured the 992 and 1856 minute gaps.
        missing = [s for s, stem in sides.items() if not os.path.exists(stem + '.sql')]
        if missing:
            bad.append((short, sides, 'missing' + ','.join(sorted(missing))))
            continue
        if '-master' not in sides:
            continue
        try:
            tm = test_time(sides['-master'] + '.sql')
        except Exception:
            continue
        for s in ('-branch', '-branch32'):
            if s not in sides:
                continue
            try:
                tb = test_time(sides[s] + '.sql')
            except Exception:
                continue
            if tm and tb and abs((tb - tm).total_seconds()) / 60 > GAP_MIN:
                bad.append((short, sides, f'gap{s}={abs((tb-tm).total_seconds())/60:.0f}m'))
                break

    print(f'  {label}: {len(bad)} cells to repair ({len(cells)} total)')
    for short, sides, gap in bad:
        files = []
        for s, stem in sorted(sides.items()):
            for ext in ('.sql', '.sql.bin', '.err'):
                if os.path.exists(stem + ext):
                    files.append(stem + ext)
                    if GO:
                        os.remove(stem + ext)
        total += len(sides)
        print(f"    {'deleted' if GO else 'would delete'} {len(sides)} sides  {gap:22s} {short}")

print(f"\n  {'deleted' if GO else 'would delete'} {total} rungs"
      f"{'' if GO else '   (re-run with --go)'}")
if GO:
    print('  next: re-run both trees; the fixed runner puts every one of these in --small')
