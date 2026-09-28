# Compare decode-time logprobs (from generation, speculative verify path when spec is on) with a clean prefill rescoring of the
# same token sequence. Usage: rescore.py <json> <index>. Aligns only if the retokenized text has the same token count.
import json, sys, urllib.request
import os
B = os.environ.get("ENDPOINT", "http://localhost:8000"); MODEL = os.environ.get("MODEL_NAME", "glm-5.3-flash")
PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_messages.json")
def post(p, b):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())
o = json.load(open(sys.argv[1]))[int(sys.argv[2])]; toks = o["tokens"]
msgs = json.load(open(PROMPT))
pids = post("/tokenize", {"model": MODEL, "messages": msgs, "add_generation_prompt": True})["tokens"]
gids = [int(t.split(":", 1)[1]) for t, _ in toks] if toks and toks[0][0].startswith("token_id:") else post("/tokenize", {"model": MODEL, "prompt": "".join(t for t, _ in toks), "add_special_tokens": False})["tokens"]
lp = post("/v1/completions", {"model": MODEL, "prompt": pids + gids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})["choices"][0]["prompt_logprobs"][len(pids):]
res = [next(iter(x.values()))["logprob"] if x else None for x in lp]
print(f"{sys.argv[1].split('/')[-1]}#{sys.argv[2]}: decode tokens {len(toks)}, retokenized {len(gids)} ({'ALIGNED' if len(gids) == len(toks) else 'not aligned: window means only'})")
for k in range(0, min(len(toks), len(res)), 2000):
    d = [l for _, l in toks[k:k + 2000]]; r = [x for x in res[k:k + 2000] if x is not None]
    line = f"   {k:5d}+: decode mean {sum(d)/len(d):.3f}  rescored mean {sum(r)/len(r):.3f}"
    if len(gids) == len(toks):
        diffs = [abs(a[1] - b) for a, b in zip(toks[k:k + 2000], res[k:k + 2000]) if b is not None]
        line += f"  mean |diff| {sum(diffs)/len(diffs):.3f}  |diff|>1 per 1k {1000*sum(x > 1 for x in diffs)/len(diffs):.0f}"
    print(line)
