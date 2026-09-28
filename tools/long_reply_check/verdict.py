# Canary verdict, rule fixed 2026-09-28 from the spec-decode hunt's measurements (rescore.py per-2,000-token windows):
# clean replies measured mean |decode - prefill| <= 0.096 and <= 5/1k tokens off by > 1 nat; broken ones >= 0.37 and >= 88/1k.
#   FAIL if any window of >= 500 tokens has mean |diff| > 0.15 or > 20/1k;  WARN if > 0.10 or > 10/1k;  else PASS.
# A reply shorter than 2,000 tokens is fine (sampling) but at least one reply must reach 4,000 tokens, else INCONCLUSIVE.
# Usage: verdict.py <rescore output file> <gen json> | --selftest
import json, re, sys
W = re.compile(r"^\s+(\d+)\+: decode mean \S+\s+rescored mean \S+\s+mean \|diff\| (\S+)\s+\|diff\|>1 per 1k (\d+)")
def verdict(lines, lens):
    rows = [(int(a), float(b), int(c)) for a, b, c in (m.groups() for m in map(W.match, lines) if m)]
    if not rows: return "INCONCLUSIVE", "no rescored windows"
    if max(lens) < 4000: return "INCONCLUSIVE", f"longest reply {max(lens)} tokens (< 4,000)"
    worst = max(rows, key=lambda r: (r[1], r[2]))
    msg = f"worst window at {worst[0]}+: mean |diff| {worst[1]:.3f}, {worst[2]}/1k > 1 nat; replies {lens} tokens"
    if any(d > 0.15 or n > 20 for _, d, n in rows): return "FAIL", msg
    if any(d > 0.10 or n > 10 for _, d, n in rows): return "WARN", msg
    return "PASS", msg
if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        ok = "    2000+: decode mean -0.6  rescored mean -0.61  mean |diff| 0.074  |diff|>1 per 1k 2"
        bad = "    4000+: decode mean -1.6  rescored mean -1.9  mean |diff| 0.387  |diff|>1 per 1k 94"
        warn = "    4000+: decode mean -1.6  rescored mean -1.7  mean |diff| 0.120  |diff|>1 per 1k 8"
        assert verdict([ok], [16000])[0] == "PASS" and verdict([ok, bad], [16000])[0] == "FAIL"
        assert verdict([ok, warn], [16000])[0] == "WARN" and verdict([ok], [300, 900])[0] == "INCONCLUSIVE" and verdict([], [16000])[0] == "INCONCLUSIVE"
        print("SELFTEST PASS"); sys.exit(0)
    lens = [len(o["tokens"]) for o in json.load(open(sys.argv[2]))]
    v, msg = verdict(open(sys.argv[1]).read().splitlines(), lens); print(v, msg)
