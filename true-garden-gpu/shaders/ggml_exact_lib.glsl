// ggml-exact arithmetic helpers for GLSL ES 3.2 compute (no #version; the
// harness prepends it). See ggml_fa_f16.comp for what they are used for.

precision highp float;
precision highp int;

// ---------------------------------------------------------------- FP16

uint f32_to_f16_rne(float f) {
    uint x = floatBitsToUint(f);
    uint sgn = (x >> 16) & 0x8000u;
    uint a = x & 0x7fffffffu;
    if (a > 0x7f800000u) return sgn | 0x7e00u;            // NaN
    if (a >= 0x477ff000u) return sgn | 0x7c00u;           // >= 65520 -> inf
    if (a >= 0x38800000u) {                               // normal half
        uint m = a - 0x38000000u;
        return sgn | ((m + 0x0fffu + ((m >> 13) & 1u)) >> 13);
    }
    uint e = a >> 23;                                     // subnormal half or zero
    if (e < 102u) return sgn;
    uint mant = (a & 0x7fffffu) | 0x800000u;
    uint sh = 126u - e;                                   // 14..24
    uint qv = mant >> sh;
    uint rem = mant & ((1u << sh) - 1u);
    uint half_ = 1u << (sh - 1u);
    if (rem > half_ || (rem == half_ && (qv & 1u) == 1u)) qv += 1u;
    return sgn | qv;
}

float f16_to_f32(uint h) {
    uint sgn = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u) {
        precise float v = float(m) * uintBitsToFloat(0x33800000u);  // m * 2^-24, exact
        return sgn != 0u ? -v : v;
    }
    if (e == 31u) return uintBitsToFloat(sgn | 0x7f800000u | (m << 13));
    return uintBitsToFloat(sgn | ((e + 112u) << 23) | (m << 13));
}

float f16r(float f) { return f16_to_f32(f32_to_f16_rne(f)); }

// ---------------------------------------------------------------- double-float helpers

vec2 two_sum(float a, float b) {
    precise float s = a + b;
    precise float bb = s - a;
    precise float e = (a - (s - bb)) + (b - bb);
    return vec2(s, e);
}

vec2 fast_two_sum(float a, float b) {
    precise float s = a + b;
    precise float e = b - (s - a);
    return vec2(s, e);
}

vec2 split(float a) {
    precise float c = 4097.0 * a;
    precise float hi = c - (c - a);
    precise float lo = a - hi;
    return vec2(hi, lo);
}

vec2 two_prod(float a, float b) {
    precise float p = a * b;
    vec2 A = split(a);
    vec2 B = split(b);
    precise float e = (((A.x * B.x - p) + A.x * B.y) + A.y * B.x) + A.y * B.y;
    return vec2(p, e);
}

vec2 ff_mul(vec2 a, vec2 b) {
    vec2 p = two_prod(a.x, b.x);
    precise float e = p.y + (a.x * b.y + a.y * b.x);
    return fast_two_sum(p.x, e);
}

vec2 ff_add(vec2 a, vec2 b) {
    vec2 s = two_sum(a.x, b.x);
    precise float e = s.y + (a.y + b.y);
    return fast_two_sum(s.x, e);
}

vec2 ffc(uint hi, uint lo) { return vec2(uintBitsToFloat(hi), uintBitsToFloat(lo)); }

// ---------------------------------------------------------------- glibc expf (generic, non-FMA build)

const uint POW2_HI[32] = uint[32](
    0x3f800000u, 0x3f82cd87u, 0x3f85aac3u, 0x3f88980fu, 0x3f8b95c2u, 0x3f8ea43au, 0x3f91c3d3u, 0x3f94f4f0u,
    0x3f9837f0u, 0x3f9b8d3au, 0x3f9ef532u, 0x3fa27043u, 0x3fa5fed7u, 0x3fa9a15bu, 0x3fad583fu, 0x3fb123f6u,
    0x3fb504f3u, 0x3fb8fbafu, 0x3fbd08a4u, 0x3fc12c4du, 0x3fc5672au, 0x3fc9b9beu, 0x3fce248cu, 0x3fd2a81eu,
    0x3fd744fdu, 0x3fdbfbb8u, 0x3fe0ccdfu, 0x3fe5b907u, 0x3feac0c7u, 0x3fefe4bau, 0x3ff5257du, 0x3ffa83b3u);
const uint POW2_LO[32] = uint[32](
    0x00000000u, 0xb34ea7a9u, 0x334f9891u, 0xb37eda4bu, 0xb260aba1u, 0xb3697465u, 0x33675624u, 0xb32e0212u,
    0x33231b71u, 0xb30c5563u, 0x33412342u, 0x30c3125au, 0xb32c9d5eu, 0xb3162b08u, 0xb22deaf6u, 0xb37c5aa8u,
    0x32cfe77au, 0x330ec5f7u, 0xb3414fe8u, 0xb2d6663eu, 0x320aa837u, 0xb37323a2u, 0x3228fc24u, 0xb35c1daau,
    0xb2d4a58au, 0xb3504a1cu, 0xb21eab59u, 0xb2441be6u, 0xb24116deu, 0xb348464au, 0x32292436u, 0xb2923758u);

float glibc_expf(float x) {
    if (x < uintBitsToFloat(0xc2cff1b4u)) return 0.0;     // also -inf
    vec2 z = ff_mul(ffc(0x4238aa3bu, 0x35257060u), vec2(x, 0.0));
    precise float k = roundEven(z.x);
    precise float t = (z.x - k) + z.y;
    bool kodd = (int(k) & 1) != 0;
    if (t > 0.5 || (t == 0.5 && kodd)) k = k + 1.0;
    else if (t < -0.5 || (t == -0.5 && kodd)) k = k - 1.0;
    precise float rh0 = z.x - k;
    vec2 r = two_sum(rh0, z.y);
    int ki = int(k);
    int i = ki & 31;
    int m = (ki - i) / 32;
    vec2 p = ff_add(ff_mul(ffc(0x35e357c2u, 0x29391239u), r), ffc(0x3975fe73u, 0xacf053b1u));
    vec2 r2 = ff_mul(r, r);
    vec2 y1 = ff_add(ff_mul(ffc(0x3cb17218u, 0xae73ad2au), r), vec2(1.0, 0.0));
    vec2 y = ff_add(ff_mul(p, r2), y1);
    y = ff_mul(y, ffc(POW2_HI[i], POW2_LO[i]));
    return ldexp(y.x, m);
}

// ---------------------------------------------------------------- correctly rounded 1/b (b > 0, normal)

float rcp_cr(float b) {
    precise float r0 = 1.0 / b;
    float best = r0;
    float best_res = 1e30;
    for (int dlt = -2; dlt <= 2; dlt++) {
        float r = uintBitsToFloat(uint(int(floatBitsToUint(r0)) + dlt));
        vec2 pe = two_prod(b, r);
        precise float res = abs((1.0 - pe.x) - pe.y);
        if (res < best_res) { best_res = res; best = r; }
    }
    return best;
}


// ---------------------------------------------------------------- correctly rounded a/b and helpers

float rcp_cr_signed(float b) {
    float r = rcp_cr(abs(b));
    return b < 0.0 ? -r : r;
}

// IEEE round-to-nearest-even a/b for normal, finite operands whose quotient is normal.
float div_cr(float a, float b) {
    precise float q0 = a / b;
    if (a == 0.0) return q0;
    float best = q0;
    float best_res = 1e30;
    for (int dlt = -3; dlt <= 3; dlt++) {
        float qc = uintBitsToFloat(uint(int(floatBitsToUint(q0)) + dlt));
        vec2 pe = two_prod(b, qc);
        precise float res = abs((a - pe.x) - pe.y);
        if (res < best_res) { best_res = res; best = qc; }
    }
    return best;
}

// int -> float with round-to-nearest-even for any 32-bit int (GLSL leaves
// the rounding of out-of-range conversions unspecified).
float i2f_rne(int v) {
    uint a = uint(v < 0 ? -v : v);
    if (a < 0x1000000u) return v < 0 ? -float(a) : float(a);
    int sh = findMSB(a) - 23;
    uint q = a >> uint(sh);
    uint rem = a & ((1u << uint(sh)) - 1u);
    uint half_ = 1u << uint(sh - 1);
    if (rem > half_ || (rem == half_ && (q & 1u) == 1u)) q += 1u;
    float f = ldexp(float(q), sh);
    return v < 0 ? -f : f;
}

// correctly rounded sqrt for positive normal a
float sqrt_cr(float a) {
    precise float s0 = sqrt(a);
    float best = s0;
    float best_res = 1e30;
    for (int dlt = -2; dlt <= 2; dlt++) {
        float r = uintBitsToFloat(uint(int(floatBitsToUint(s0)) + dlt));
        vec2 pe = two_prod(r, r);
        precise float res = abs((a - pe.x) - pe.y);
        if (res < best_res) { best_res = res; best = r; }
    }
    return best;
}

// ggml_v_expf, SSE2 build without FMA (MADD128 = mul then add), per lane.
float ggml_v_expf(float x) {
    const float r = 12582912.0;  // 0x1.8p23
    precise float z = x * uintBitsToFloat(0x3fb8aa3bu) + r;      // 0x1.715476p+0
    precise float n = z - r;
    precise float b = (x - n * uintBitsToFloat(0x3f317200u)) - n * uintBitsToFloat(0x35bfbe8eu);  // 0x1.62e4p-1, 0x1.7f7d1cp-20
    uint e = floatBitsToUint(z) << 23;
    float k = uintBitsToFloat(e + floatBitsToUint(1.0));
    bool c = abs(n) > 126.0;
    precise float u = b * b;
    precise float j = ((uintBitsToFloat(0x3c072010u) * b + uintBitsToFloat(0x3d2b9f17u)) * u
                       + (uintBitsToFloat(0x3e2aaf33u) * b + uintBitsToFloat(0x3efffedbu))) * u
                      + uintBitsToFloat(0x3f7ffff6u) * b;   // 0x1.0e4020p-7, 0x1.573e2ep-5, 0x1.555e66p-3, 0x1.fffdb6p-2, 0x1.ffffecp-1
    if (!c) {
        precise float res = j * k + k;
        return res;
    }
    uint g = (n <= 0.0) ? 0x82000000u : 0u;
    float s1 = uintBitsToFloat(g + 0x7f000000u);
    float s2 = uintBitsToFloat(e - g);
    if (abs(n) > 192.0) {
        precise float res = s1 * s1;
        return res;
    }
    precise float res = (s2 * j + s2) * s1;
    return res;
}
