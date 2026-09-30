#!/usr/bin/env python3
# Builds the master-vs-fused ratio table from the two llama-bench -o sql dumps.
import sqlite3, re, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(":memory:")
for tag in ("master", "fused"):
    sql = open(f"{tag}.sql").read()
    # keep one CREATE TABLE, tag each INSERT with the side
    if "llama_bench" not in [r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")]:
        m = re.search(r"CREATE TABLE[^;]+;", sql)
        con.execute(m.group(0))
        con.execute("alter table llama_bench add column side text")
    for ins in re.findall(r"INSERT INTO llama_bench[^;]+;", sql):
        con.execute(ins.replace(");", f", '{tag}');").replace(
            ") VALUES", ", side) VALUES"))

rows = con.execute("""
    select type_k, n_ubatch, n_prompt, n_gen, n_depth, side, avg_ts, stddev_ts
    from llama_bench order by type_k, n_prompt desc, n_ubatch, n_depth
""").fetchall()
pair = {}
for tk, ub, npp, ng, nd, side, ts, sd in rows:
    pair.setdefault((tk, ub, npp, ng, nd), {})[side] = (ts, sd)

print(f"{'type_kv':>7} {'ub':>4} {'test':>15} {'master t/s':>12} {'fused t/s':>12} {'ratio':>6}")
for (tk, ub, npp, ng, nd), d in pair.items():
    test = (f"pp{npp}" if npp else f"tg{ng}") + f"@d{nd}"
    m = d.get("master"); f = d.get("fused")
    if not (m and f):
        print(f"{tk:>7} {ub:>4} {test:>15}   MISSING SIDE {list(d)}")
        continue
    print(f"{tk:>7} {ub:>4} {test:>15} {m[0]:>12.2f} {f[0]:>12.2f} {f[0]/m[0]:>6.3f}")
