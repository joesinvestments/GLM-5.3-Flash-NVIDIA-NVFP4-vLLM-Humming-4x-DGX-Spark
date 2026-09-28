# Corruption probe for the vLLM #54150 symptom (ModelOpt NVFP4: invalid UTF-8 byte tokens -> U+FFFD in text).
# 6 multilingual prompts x 2 runs (greedy, temperature 1.0), 700 tokens each, thinking off. Counts U+FFFD, lone surrogates and
# control characters in the output. Usage: ENDPOINT=http://head:8000 corruption_probe.py <label> <out.json>   Prints one line: PROBE <label> fffd=<n> ...
import concurrent.futures as cf, json, os, sys, time, unicodedata, urllib.request
B = os.environ.get("ENDPOINT", "http://localhost:8000"); MODEL = os.environ.get("MODEL_NAME", "glm-5.3-flash"); LABEL, OUT = sys.argv[1], sys.argv[2]
P = {
 "korean": "한국어로 서울의 역사와 오늘날의 문화에 대해 자세히 설명해 주세요. 음식, 음악, 기술 산업을 포함해 주세요.",
 "chinese": "请用中文详细介绍长城的历史、建造过程以及它在中国文化中的意义。",
 "japanese": "日本語で、京都の寺院と四季の行事について詳しく説明してください。",
 "russian": "Подробно расскажи на русском языке об истории Санкт-Петербурга и его архитектуре.",
 "arabic": "اشرح بالتفصيل باللغة العربية تاريخ مدينة القاهرة وأهم معالمها.",
 "emoji_mix": "Write a cheerful travel guide for Tokyo, Seoul and Paris. Use lots of emoji 🍜🗼🎌🥐✨ in every paragraph and include one sentence in Korean, one in Japanese and one in French.",
}
def run(args):
    k, temp = args
    body = {"model": MODEL, "messages": [{"role": "user", "content": P[k]}], "max_tokens": 700, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": False}, **({"seed": 7} if temp else {})}
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read()); t = d["choices"][0]["message"].get("content") or ""
    bad_ctrl = sum(1 for ch in t if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r")
    return {"prompt": k, "temp": temp, "tokens": d["usage"]["completion_tokens"], "fffd": t.count("�"),
            "surrogates": sum(1 for ch in t if 0xD800 <= ord(ch) <= 0xDFFF), "ctrl": bad_ctrl, "chars": len(t), "text": t}
with cf.ThreadPoolExecutor(6) as ex: res = list(ex.map(run, [(k, t) for k in P for t in (0, 1.0)]))
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
f = sum(r["fffd"] for r in res); s = sum(r["surrogates"] for r in res); c = sum(r["ctrl"] for r in res)
print(f"PROBE {LABEL}: fffd={f} surrogates={s} ctrl={c} over {sum(r['chars'] for r in res)} chars / {sum(r['tokens'] for r in res)} tokens, "
      f"12 replies; per prompt fffd: " + " ".join(f"{r['prompt']}@{r['temp']}={r['fffd']}" for r in res if r["fffd"]))
