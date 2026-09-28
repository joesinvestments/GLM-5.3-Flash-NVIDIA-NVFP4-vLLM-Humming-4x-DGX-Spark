# Long-reply check

Catches speculative-decoding output corruption that short evals miss. It is how we found that an 8-bit DFlash2 drafter
(patch 10) made long replies collapse.

1. `gen.py out.json`: two replies of up to 16,000 tokens (thinking at max, temperature 1.0, unique `cache_salt`) with the
   decode-time logprob and exact token id of every generated token.
2. `rescore.py out.json 0` and `... 1`: feeds prompt + reply back as a prompt (`prompt_logprobs`) and compares each token's
   decode-time logprob with the prefill logprob of the same token id, per 2,000-token window.
3. `verdict.py rescore.txt out.json`: FAIL if any window has mean |decode - prefill| > 0.15 or more than 20 tokens per
   1,000 off by more than 1 nat; WARN above 0.10 or 10 per 1,000.

Healthy decoding agrees with prefill closely (we measured 0.05 to 0.10 mean |diff| with speculation off and with the 16-bit
drafter); the broken 8-bit drafter showed 0.37 to 1.0 and up to 269 tokens per 1,000 off, starting a few thousand tokens in.
Set `ENDPOINT` (default http://localhost:8000) and `MODEL_NAME` (default glm-5.3-flash). The prompt is a synthetic coding
task; two local paths in its tool outputs were replaced with placeholders.
