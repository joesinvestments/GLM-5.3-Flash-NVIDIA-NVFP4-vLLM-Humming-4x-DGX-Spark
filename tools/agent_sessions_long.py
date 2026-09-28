# Long-context agent sessions, all thinking at max: the load behind the README's C8 numbers. Same sessions, tools and canned
# tool results as agent_sessions.py, at the context real agent traffic has (~19K-token prompts): system prompt + tools + a
# shared project reference (~12K tokens, identical for every session), then per task a unique session line and ~8K tokens of
# files, so sessions share only the system part. Every request sends reasoning_effort max. Per level, from the server's own
# counters: aggregate tok/s, draft acceptance, mean TTFT, prefix-cache hit rate, prompt size and foreign traffic.
# The text pool is the vllm/v1 tree of a vLLM v0.30.0 checkout (VLLM_V1_SRC). Compare runs only at similar hit rates: a warm
# cache (96% hits) read 1.55x a cold one (84%) on the same build.
# Usage: ENDPOINT=http://head:8000 VLLM_V1_SRC=/path/to/vllm-v0.30.0/vllm/v1 [LEVELS_ONLY=8] python3 agent_sessions_long.py <label> <out.json> [sweeps]
import concurrent.futures as cf, glob, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_sessions as al

TREE = os.environ.get("VLLM_V1_SRC", "vllm/v1")
POOL = "".join(f"\n### file: vllm/v1/{os.path.relpath(p, TREE)}\n" + open(p).read() for p in sorted(glob.glob(TREE + "/**/*.py", recursive=True)))
SHORT = os.environ.get("SHORT") == "1"  # short agent turns (~1.5-2.5K-token prompts): what the 2,304-token cache block never resumes
SHARED_CHARS, SLICE_CHARS, NSLICES = (2000, 2400, 24) if SHORT else (48000, 32000, 24)
MAXTOK = int(os.environ.get("MAXTOK", "1536"))  # reply cap per turn (reasoning + answer)
assert len(POOL) >= SHARED_CHARS + NSLICES * SLICE_CHARS, len(POOL)
SYSTEM = al.SYSTEM + "\n\n# Project reference (loaded for every session)\n" + POOL[:SHARED_CHARS]
LEVELS = {1: 4, 2: 3, 4: 2, 8: 2}  # concurrent sessions: tasks per session
if os.environ.get("LEVELS_ONLY"):  # e.g. LEVELS_ONLY=8; unset = 1, 2, 4 and 8 sessions
    LEVELS = {int(c): LEVELS[int(c)] for c in os.environ["LEVELS_ONLY"].split(",")}
NAMES = {"acc": "vllm:spec_decode_num_accepted_tokens_total", "draft": "vllm:spec_decode_num_draft_tokens_total",
         "req": "vllm:request_success_total", "gen": "vllm:generation_tokens_total", "prompt": "vllm:prompt_tokens_total",
         "hits": "vllm:prefix_cache_hits_total", "queries": "vllm:prefix_cache_queries_total",
         "ttft_sum": "vllm:time_to_first_token_seconds_sum", "ttft_n": "vllm:time_to_first_token_seconds_count"}

def first_user(nonce, k, user):
    lo = SHARED_CHARS + (k % NSLICES) * SLICE_CHARS
    return f"[session {nonce}]\n# Files you opened earlier in this session\n{POOL[lo:lo + SLICE_CHARS]}\n\n# Task\n{user}"

def metrics():  # exact names (a prefix match would also sum histogram buckets), summed over label sets
    m, by = dict.fromkeys(NAMES, 0.0), {v: k for k, v in NAMES.items()}
    for line in urllib.request.urlopen(al.B + "/metrics", timeout=10).read().decode().splitlines():
        if line.startswith("#") or " " not in line: continue
        name, val = line.rsplit(" ", 1); k = by.get(name.split("{")[0])
        if k: m[k] += float(val)
    return m

def task(ti, think, seed, nonce):  # agent_load.task with the long first message; also records prompt size
    user, result_a, result_b = al.TASKS[ti % len(al.TASKS)]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first_user(nonce, ti, user)}]; out = []
    for turn in range(3):
        body = {"model": al.MODEL, "messages": msgs, "tools": al.TOOLS, "max_tokens": MAXTOK, "seed": seed + turn,
                "reasoning_effort": "max"}  # all-thinking, as agent clients with maximum reasoning send
        t0 = time.time(); d = al.post(body); dt = time.time() - t0
        m = d["choices"][0]["message"]; calls = m.get("tool_calls") or []
        out.append({"task": ti, "turn": turn, "think": think, "tokens": d["usage"]["completion_tokens"], "prompt": d["usage"]["prompt_tokens"],
                    "s": round(dt, 2), "finish": d["choices"][0]["finish_reason"], "tool_calls": len(calls)})
        msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
        canned = result_a if turn == 0 else result_b
        if calls:
            for i, c in enumerate(calls): msgs.append({"role": "tool", "tool_call_id": c["id"], "content": canned if i == 0 else "ok"})
        else:
            msgs.append({"role": "user", "content": "Here is what the tool returned:\n" + canned + "\nContinue."})
    return out

def level(c, n_tasks, sweep):
    def session(s):
        think = True  # all sessions think
        res = []
        for k in range(n_tasks):
            res += task(s * 3 + k + sweep, True, 1000 * sweep + 100 * s + 10 * k, f"c{c}-w{sweep}-s{s}-k{k}")
        return res
    m0 = metrics(); t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex: reqs = [r for rs in ex.map(session, range(c)) for r in rs]
    wall = time.time() - t0; time.sleep(0.5); m1 = metrics(); d = {k: m1[k] - m0[k] for k in m0}; tok = sum(r["tokens"] for r in reqs)
    return {"c": c, "sweep": sweep, "wall_s": round(wall, 1), "requests": len(reqs), "tokens": tok, "agg_tok_s": round(tok / wall, 1),
            "acceptance": round(d["acc"] / d["draft"], 4) if d["draft"] > 0 else None,
            "foreign_requests": int(d["req"]) - len(reqs), "foreign_tokens": int(d["gen"]) - tok,
            "prompt_tok_mean": round(sum(r["prompt"] for r in reqs) / len(reqs)),
            "ttft_mean_s": round(d["ttft_sum"] / d["ttft_n"], 3) if d["ttft_n"] else None,
            "hit_rate": round(d["hits"] / d["queries"], 4) if d["queries"] else 0.0,
            "tool_call_turns": sum(r["tool_calls"] > 0 for r in reqs), "length_cut": sum(r["finish"] == "length" for r in reqs), "reqs": reqs}

if __name__ == "__main__":
    if "--dry" in sys.argv:
        print(f"pool {len(POOL)} chars, system {len(SYSTEM)} chars, first user {len(first_user('x', 0, al.TASKS[0][0]))} chars"); sys.exit(0)
    res = {"label": al.LABEL, "t0": time.strftime("%FT%T"), "levels": []}
    task(0, False, 1, "warmup")  # warm-up, not measured; its unique session line keeps it out of the measured prefixes
    for sweep in range(al.SWEEPS):
        for c, n in LEVELS.items():
            r = level(c, n, sweep); res["levels"].append(r); json.dump(res, open(al.OUT, "w"), indent=1)
            print(f"{al.LABEL} sweep {sweep} C{c}: {r['agg_tok_s']} tok/s ttft {r['ttft_mean_s']} s hit {r['hit_rate']} prompt {r['prompt_tok_mean']} "
                  f"({r['tokens']} tok / {r['wall_s']} s, {r['requests']} req) acc {r['acceptance']} length-cut {r['length_cut']} "
                  f"foreign {r['foreign_requests']}/{r['foreign_tokens']}", flush=True)
