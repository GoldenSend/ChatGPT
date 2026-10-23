#!/usr/bin/env python3
"""Compare GPU-022 replay outputs with the bundle's expected outputs, using the
standard library only (the PS4 needs no numpy). Values are compared as float32
bit patterns; two NaNs count as equal whatever their payload.

usage: gate_compare.py OUTDIR EXPECTEDDIR LABELS.tsv PART [MAXSTEP]
Exit 0 when every output of PART (up to decode step MAXSTEP for part C) matches.
"""
import math
import os
import struct
import sys
from array import array


def ndiff(a, b):
    if a == b:
        return 0, -1
    if len(a) != len(b) or len(a) % 4:
        return max(len(a), len(b)) // 4, 0
    x, y = array("I", a), array("I", b)
    n, first = 0, -1
    for i in range(len(x)):
        if x[i] != y[i]:
            fx, fy = struct.unpack("<ff", struct.pack("<II", x[i], y[i]))
            if not (math.isnan(fx) and math.isnan(fy)):
                n += 1
                first = i if first < 0 else first
    return n, first


def main():
    out, exp, labels, part = sys.argv[1:5]
    maxstep = int(sys.argv[5]) if len(sys.argv) > 5 else 1 << 30
    checked = bad = 0
    first_bad = None
    for line in open(labels):
        label, p, step, stage, _ = line.rstrip("\n").split("\t")
        if p != part or int(step) > maxstep:
            continue
        po = os.path.join(out, label + ".bin")
        if not os.path.exists(po):
            print(f"MISSING {label}")
            bad += 1
            first_bad = first_bad or label
            continue
        n, i = ndiff(open(po, "rb").read(), open(os.path.join(exp, label + ".bin"), "rb").read())
        checked += 1
        if n:
            bad += 1
            first_bad = first_bad or label
            print(f"DIFF {label} (stage {stage}, step {step}): {n} values differ, first at index {i}")
    if bad:
        print(f"COMPARE {part}: FAIL, {bad} of {checked} outputs differ; first: {first_bad}")
    else:
        print(f"COMPARE {part}: PASS, {checked} outputs bit-exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
