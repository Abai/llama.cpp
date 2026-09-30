#!/usr/bin/env python3
"""Helper for scripts/sass-diff-fattn.sh.

Subcommands:
  commands <compile_commands.json> <glob>
      Print (one per line) shell commands that compile the
      fattn-mma-f16-instance-<glob>.cu translation units, with a
      `mkdir -p` for the object dir and a `cd` into the entry dir.
  diff <A.o> <B.o>
      Dump SASS from both objects, normalize, pair kernels by order of
      appearance, diff. Prints resource-usage table and any instruction
      divergence. Exit 0 if identical, 1 if not.
"""

import difflib
import fnmatch
import json
import re
import shlex
import shutil
import subprocess
import sys

ADDR_RE   = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s*")
IMM_RE    = re.compile(r"0x[0-9a-fA-F]+")
# decimal float immediates: sm_120 stubs materialize __LINE__ as an fp16 subnormal via HFMA2
FLT_RE    = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
# instructions that only appear in kernels doing real FA work (smem/mma/async-copy)
WORK_RE   = re.compile(r"\b(HMMA|IMMA|LDSM|LDGSTS|LDS|STS)\b")
ENC_RE    = re.compile(r"/\*\s*0x[0-9a-fA-F]+\s*\*/")
LABEL_RE  = re.compile(r"\.L_x_\d+|\.L_\d+")
FUNC_RE   = re.compile(r"^\s*Function\s*:\s*(\S+)")
RES_FN_RE = re.compile(r"^\s*Function (\S+?):?$")


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def demangle(names):
    tool = shutil.which("cu++filt") or shutil.which("c++filt")
    if not tool:
        return names
    out = subprocess.run([tool], input="\n".join(names) + "\n",
                         capture_output=True, text=True).stdout.splitlines()
    return out if len(out) == len(names) else names


def parse_sass(obj):
    """Return list of (mangled_name, [normalized instruction lines])."""
    text = run(["cuobjdump", "--dump-sass", obj])
    funcs = []
    cur = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            cur = []
            funcs.append((m.group(1), cur))
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s or s.startswith(".headerflags") or set(s) == {"."}:
            continue
        if s.startswith(("Fatbin", "arch =", "code version", "host =",
                         "compile_size", "code for", "compressed", "identifier")):
            cur = None
            continue
        line = ADDR_RE.sub("", line)
        line = ENC_RE.sub("", line)
        line = " ".join(line.split())
        if line:
            cur.append(line)
    # normalize branch labels positionally per function
    out = []
    for name, body in funcs:
        seen = {}

        def sub(m):
            tok = m.group(0)
            if tok not in seen:
                seen[tok] = ".LBL%d" % len(seen)
            return seen[tok]

        out.append((name, [LABEL_RE.sub(sub, ln) for ln in body]))
    return out


def parse_resources(obj):
    """Return list of (mangled_name, resource string)."""
    text = run(["cuobjdump", "--dump-resource-usage", obj])
    res = []
    name = None
    for line in text.splitlines():
        m = RES_FN_RE.match(line)
        if m:
            name = m.group(1)
            continue
        if name and ("REG:" in line or "STACK:" in line):
            res.append((name, " ".join(line.split())))
            name = None
    return res


def cmd_commands(cc_path, glob):
    pat = "*/template-instances/fattn-mma-f16-instance-%s.cu" % glob
    entries = [e for e in json.load(open(cc_path))
               if fnmatch.fnmatch(e["file"], pat)]
    if not entries:
        return 1
    for e in entries:
        cmd = e.get("command") or " ".join(shlex.quote(a) for a in e["arguments"])
        out = e.get("output", "")
        pre = ""
        if out:
            pre = "mkdir -p %s && " % shlex.quote(out.rsplit("/", 1)[0]) if "/" in out else ""
        print("cd %s && %s%s" % (shlex.quote(e["directory"]), pre, cmd))
    return 0


def cmd_diff(obj_a, obj_b):
    rc = 0
    res_a, res_b = parse_resources(obj_a), parse_resources(obj_b)
    sass_a, sass_b = parse_sass(obj_a), parse_sass(obj_b)

    if len(sass_a) != len(sass_b):
        print("  kernel count mismatch: %d vs %d" % (len(sass_a), len(sass_b)))
        rc = 1

    names_a = demangle([n for n, _ in sass_a])
    names_b = demangle([n for n, _ in sass_b])

    # fast path: resource usage (REG/STACK/SHARED/LOCAL...)
    for k, ((_, ra), (_, rb)) in enumerate(zip(res_a, res_b)):
        short = names_a[k] if k < len(names_a) else "kernel%d" % k
        if ra != rb:
            print("  RES-DIFF kernel %d %s" % (k, short))
            print("    A: %s" % ra)
            print("    B: %s" % rb)
            rc = 1
        else:
            print("  res kernel %-2d %-60s %s" % (k, short[:60], ra))

    for k, ((_, body_a), (_, body_b)) in enumerate(zip(sass_a, sass_b)):
        if body_a == body_b:
            continue
        # NO_DEVICE_CODE stubs (skipped variants) embed __FILE__/__LINE__/__FUNCTION__,
        # which legitimately shift across refactors. A kernel with no smem/mma/async-copy
        # instructions does no FA work, so mask immediates there and compare the rest.
        stub = not any(WORK_RE.search(ln) for ln in body_a + body_b)
        if stub:
            body_a = [FLT_RE.sub("FIMM", IMM_RE.sub("0xIMM", ln)) for ln in body_a]
            body_b = [FLT_RE.sub("FIMM", IMM_RE.sub("0xIMM", ln)) for ln in body_b]
            if body_a == body_b:
                print("  note kernel %-2d stub (NO_DEVICE_CODE): identical after masking source-location immediates" % k)
                continue
        rc = 1
        print("  SASS-DIFF kernel %d%s (%d vs %d instructions)" % (k, " [stub]" if stub else "", len(body_a), len(body_b)))
        print("    A: %s" % names_a[k])
        print("    B: %s" % names_b[k])
        d = list(difflib.unified_diff(body_a, body_b, "A", "B", lineterm="", n=1))
        for ln in d[:40]:
            print("    " + ln)
        if len(d) > 40:
            print("    ... (%d more diff lines)" % (len(d) - 40))
    return rc


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if sys.argv[1] == "commands":
        return cmd_commands(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "diff":
        return cmd_diff(sys.argv[2], sys.argv[3])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
