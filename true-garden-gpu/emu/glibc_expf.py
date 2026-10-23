"""Reference model of glibc's expf (sysdeps/ieee754/flt-32/e_expf.c, glibc 2.28+).

x86-64 glibc selects at run time (ifunc) between the generic build and an
FMA build (e_expf-fma.c, used on CPUs with FMA/AVX2). AMD Jaguar has no FMA,
so the PS4 uses the generic one; most modern x86 hosts (IdeaPad, this cloud
container) use the FMA one. The two agree except in rare near-halfway cases.
"""
import math
import struct

import numpy as np

N = 32
TAB = [
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
]
SHIFT = float.fromhex("0x1.8p+52")
INVLN2N = float.fromhex("0x1.71547652b82fep+0") * N
C0 = float.fromhex("0x1.c6af84b912394p-5") / N / N / N
C1 = float.fromhex("0x1.ebfce50fac4f3p-3") / N / N
C2 = float.fromhex("0x1.62e42ff0c52d6p-1") / N


def _scale(ki):
    t = (TAB[ki % N] + ((ki << (52 - 5)) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return struct.unpack("<d", struct.pack("<Q", t))[0]


def expf_generic(x):
    """Non-FMA build, valid for |x| < 88 (attention arguments are <= 0 and > -88 in practice)."""
    xd = float(np.float32(x))
    z = INVLN2N * xd
    kd = z + SHIFT
    ki = struct.unpack("<Q", struct.pack("<d", kd))[0]
    kd -= SHIFT
    r = z - kd
    s = _scale(ki)
    zz = C0 * r + C1
    r2 = r * r
    y = C2 * r + 1
    y = zz * r2 + y
    y = y * s
    return np.float32(y)


def expf_fma(x):
    """FMA build: the three polynomial multiply-adds are fused (model; see TESTS for the check against libm)."""
    xd = float(np.float32(x))
    z = INVLN2N * xd
    kd = z + SHIFT
    ki = struct.unpack("<Q", struct.pack("<d", kd))[0]
    kd -= SHIFT
    r = z - kd
    s = _scale(ki)
    zz = math.fma(C0, r, C1)
    r2 = r * r
    y = math.fma(C2, r, 1.0)
    y = math.fma(zz, r2, y)
    y = y * s
    return np.float32(y)


# --------------------------------------------------------------------------
# float32-only ("double-float") port of expf_generic, as used by the GLSL
# shader (GLES has no fp64). Mirrors shaders/ggml_fa_f16.comp line by line.

F = np.float32


def _two_sum(a, b):
    s = F(a + b)
    bb = F(s - a)
    return s, F(F(a - F(s - bb)) + F(b - bb))


def _fast_two_sum(a, b):
    s = F(a + b)
    return s, F(b - F(s - a))


def _split(a):
    c = F(F(4097.0) * a)
    hi = F(c - F(c - a))
    return hi, F(a - hi)


def _two_prod(a, b):
    p = F(a * b)
    ah, al = _split(a)
    bh, bl = _split(b)
    e = F(F(F(F(F(ah * bh) - p) + F(ah * bl)) + F(al * bh)) + F(al * bl))
    return p, e


def _ff_mul(a, b):
    p, e = _two_prod(a[0], b[0])
    e = F(e + F(F(a[0] * b[1]) + F(a[1] * b[0])))
    return _fast_two_sum(p, e)


def _ff_add(a, b):
    s, e = _two_sum(a[0], b[0])
    e = F(e + F(a[1] + b[1]))
    return _fast_two_sum(s, e)


def _ff_const(v):
    hi = F(v)
    return (hi, F(v - float(hi)))


INVLN2N_FF = _ff_const(INVLN2N)
C0_FF, C1_FF, C2_FF = _ff_const(C0), _ff_const(C1), _ff_const(C2)
POW2_FF = [_ff_const(struct.unpack("<d", struct.pack("<Q", TAB[i] + (i << 47)))[0]) for i in range(N)]


def expf_ff(x):
    x = F(x)
    if x == F(-np.inf) or x < F(float.fromhex("-0x1.9fe368p6")):
        return F(0.0)
    zh, zl = _ff_mul(INVLN2N_FF, (x, F(0.0)))
    k = F(np.round(zh))  # roundEven
    rh = F(zh - k)
    t = F(rh + zl)
    if t > F(0.5) or (t == F(0.5) and (int(k) & 1)):
        k = F(k + F(1.0))
    elif t < F(-0.5) or (t == F(-0.5) and (int(k) & 1)):
        k = F(k - F(1.0))
    rh, rl = _two_sum(F(zh - k), zl)
    ki = int(k)
    i, m = ki % N, ki // N
    r = (rh, rl)
    p = _ff_add(_ff_mul(C0_FF, r), C1_FF)
    r2 = _ff_mul(r, r)
    y1 = _ff_add(_ff_mul(C2_FF, r), (F(1.0), F(0.0)))
    y = _ff_add(_ff_mul(p, r2), y1)
    y = _ff_mul(y, POW2_FF[i])
    return F(np.ldexp(y[0], m))


def expf_generic_vec(x):
    """Vectorized expf_generic (numpy float64 ops are unfused, like the non-FMA build). |x| < 88."""
    xd = np.asarray(x, np.float32).astype(np.float64)
    z = INVLN2N * xd
    kd = z + SHIFT
    ki = kd.view(np.uint64)
    kd = kd - SHIFT
    r = z - kd
    tab = np.array(TAB, np.uint64)
    t = tab[(ki % np.uint64(N)).astype(np.int64)] + (ki << np.uint64(52 - 5))
    s = t.view(np.float64)
    zz = C0 * r + C1
    r2 = r * r
    y = C2 * r + 1
    y = zz * r2 + y
    y = y * s
    return y.astype(np.float32)
