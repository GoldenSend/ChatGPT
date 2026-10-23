#!/usr/bin/env python3
"""Turn RESULTS/sensitivity.json into a markdown table. usage: summarize_sensitivity.py IN.json OUT.md"""
import collections
import json
import statistics
import sys


def main():
    r = json.load(open(sys.argv[1]))
    out = ["| Variant (one op computed GPU-style) | token0 fails | token1 fails | token1 first differing stage | "
           "token1 Q8_K flips at gate/up input (median) | token1 max abs pre-FFN diff | token1 bad l_out @1e-5 (median of fails) |",
           "|---|---|---|---|---|---|---|"]
    for name, recs in r["variants"].items():
        t0 = sum(x["token0"]["l_out_bad"]["atol=1e-05,rtol=0"] > 0 for x in recs)
        t1b = [x["token1"]["l_out_bad"]["atol=1e-05,rtol=0"] for x in recs]
        t1 = sum(b > 0 for b in t1b)
        first = collections.Counter(x["token1"]["first_diff"] or "-" for x in recs).most_common(1)[0][0]
        flips = statistics.median(x["token1"]["q8_flips"]["ffn_norm"] for x in recs)
        pre = max(x["token1"]["pre_ffn_maxabs"] for x in recs)
        fails = [b for b in t1b if b]
        out.append(f"| {name} | {t0}/{len(recs)} | {t1}/{len(recs)} | {first} | {flips:g} | {pre:.2e} | "
                   f"{statistics.median(fails) if fails else 0:g} |")
    out.append("")
    out.append("Single Q8_K code flip probe (token 1, exact pipeline, one `ffn_norm` element moved across its rounding boundary):")
    out.append("")
    out.append("| model | input change | codes changed | downstream Q8_K flips at down-proj input | l_out values changed | l_out max abs | bad @1e-5 |")
    out.append("|---|---|---|---|---|---|---|")
    for sf in r["single_flip"]:
        p = sf["nearest-boundary element"]
        out.append(f"| {sf['model']} | {p['input_change_abs']:.2e} | {p['q8_codes_changed']} | "
                   f"{p['gate_up_down_q8_flips_downstream']} | {p['l_out_ndiff']}/1024 | {p['l_out_maxabs']:.2e} | {p['l_out_bad_atol_1e_5']} |")
    open(sys.argv[2], "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
