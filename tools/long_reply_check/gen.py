# Long-reply check, step 1: 2 long replies (16K cap, thinking max, temperature 1.0, unique cache_salt) with exact token ids and
# decode-time logprobs, from a ~1.3K-token coding-agent prompt (prompt_messages.json). Usage: gen.py <out.json>
import concurrent.futures as cf, json, sys, time, urllib.request
import os
B = os.environ.get("ENDPOINT", "http://localhost:8000"); MODEL = os.environ.get("MODEL_NAME", "glm-5.3-flash")
PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_messages.json")
msgs = json.load(open(PROMPT))
def post(path, body):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
def gen(i):
    t = time.time()
    r = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "reasoning_effort": "max", "temperature": 1.0, "max_tokens": 16000,
             "cache_salt": f"canary-{time.time()}-{i}", "logprobs": True, "top_logprobs": 1, "return_tokens_as_token_ids": True})
    c = r["choices"][0]; m = c["message"]; lp = (c.get("logprobs") or {}).get("content") or []
    return {"i": i, "secs": round(time.time() - t, 1), "usage": r["usage"], "finish": c["finish_reason"],
            "content_chars": len(m.get("content") or ""), "tokens": [[x["token"], round(x["logprob"], 3)] for x in lp]}
with cf.ThreadPoolExecutor(2) as ex: outs = list(ex.map(gen, range(2)))
json.dump(outs, open(sys.argv[1], "w"))
for o in outs: print(f"#{o['i']}: {o['usage']['completion_tokens']} tokens in {o['secs']} s, finish {o['finish']}")
