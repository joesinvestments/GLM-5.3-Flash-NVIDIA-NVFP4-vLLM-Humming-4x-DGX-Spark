# Checks a ModelOpt NVFP4 MoE checkpoint on Hugging Face for the vLLM #54150 defect without downloading it: vLLM's fused
# gate/up (w13) repack uses ONE global scale per expert (the gate's), so an expert whose gate_proj and up_proj
# weight_scale_2 differ gets its up half mis-scaled, which shows up as corrupted tokens. Reads the safetensors headers and a
# few bytes per scale with HTTP range requests.
# Usage: python3 check_gate_up_scales.py <repo> <revision> [experts per layer, default 8]    (HF_TOKEN for gated repos)
import collections, json, os, random, re, struct, sys, urllib.request
REPO, REV = sys.argv[1], sys.argv[2]; PER = int(sys.argv[3]) if len(sys.argv) > 3 else 8
H = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
def url(f): return f"https://huggingface.co/{REPO}/resolve/{REV}/{f}"
def get(f, a=None, b=None):
    h = dict(H); a is not None and h.update(Range=f"bytes={a}-{b}")
    return urllib.request.urlopen(urllib.request.Request(url(f), headers=h), timeout=60).read()
idx = json.loads(get("model.safetensors.index.json"))["weight_map"]
pat = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj)\.weight_scale_2$")
by = collections.defaultdict(dict)
for k in idx:
    m = pat.search(k)
    if m: by[(int(m.group(1)), int(m.group(2)))][m.group(3)] = k
if not by: sys.exit("no per-expert gate_proj/up_proj weight_scale_2 tensors found (not a ModelOpt per-expert NVFP4 layout)")
hdr = {}
def val(k):
    f = idx[k]
    if f not in hdr:
        n = struct.unpack("<Q", get(f, 0, 7))[0]; hdr[f] = (n, json.loads(get(f, 8, 7 + n)))
    n, h = hdr[f]; m = h[k]; a, _ = m["data_offsets"]; raw = get(f, 8 + n + a, 8 + n + a + 3)
    return {"F32": lambda: struct.unpack("<f", raw)[0], "BF16": lambda: struct.unpack("<f", b"\0\0" + raw[:2])[0],
            "F16": lambda: struct.unpack("<e", raw[:2])[0]}[m["dtype"]]()
layers = sorted({L for L, _ in by}); tot = mis = 0; worst = 1.0
for L in layers:
    ex = sorted(e for (l, e) in by if l == L)
    for e in random.Random(L).sample(ex, min(PER, len(ex))):
        g, u = val(by[(L, e)]["gate_proj"]), val(by[(L, e)]["up_proj"]); tot += 1; mis += g != u; worst = max(worst, max(g, u) / min(g, u))
print(f"{REPO}@{REV[:10]}: {tot} expert pairs sampled over {len(layers)} MoE layers; gate != up global scale in {mis} "
      f"({100 * mis / tot:.1f}%), worst ratio {worst:.3f}. {'Not affected by' if mis == 0 else 'AFFECTED by'} the vLLM #54150 single-scale repack.")
