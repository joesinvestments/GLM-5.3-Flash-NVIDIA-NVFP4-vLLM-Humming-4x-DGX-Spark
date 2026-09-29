# GLM-5.3-Flash on 4x DGX Spark: NVIDIA NVFP4, vLLM main, Humming

GLM-5.3-Flash served across four NVIDIA DGX Sparks (GB10, tensor parallel 4 over RoCE) with:

- **Weights:** NVIDIA's official [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) at revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` (MIT): ModelOpt 0.47 NVFP4 on the routed experts and the dense MLP of layers 0 to 2; attention and shared experts in BF16.
- **Engine:** vLLM main, nightly `7f1a5398e9` (0.30.1rc1.dev48), not a release, plus our 19-patch series for GB10 and this model, including upstream fixes we carry ahead of vLLM.
- **Kernels:** [Humming](https://github.com/vllm-project/humming) for the MoE (`--moe-backend humming`), Marlin for the NVFP4 dense layers (`--linear-backend marlin`).
- **Speculative decoding:** the [incoai DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) in 16-bit, standard verification, up to 7 draft tokens.
- **Context:** 500,000 tokens, fp8 KV cache (35 GiB per rank).

This is the exact build we serve: `image-main/build.sh` refuses to build unless every patched file matches the checksums of our production image, and `launch/launch_node.sh` carries our production serving flags.

## Results (2026-09-28)

All numbers are from the production cluster, measured on the day. Load and long-reply numbers were measured with thinking at max, the heaviest setting; the recommended client setting is high (finding 6).

| What | How it was measured | Result |
|---|---|---|
| Throughput, 8 concurrent agent sessions | `tools/agent_sessions_long.py`, `LEVELS_ONLY=8`, 3 sweeps: 3-turn tool-calling tasks, ~19K-token prompts, thinking at max, prefix-cache hit rate 0.83 to 0.85 | **54.9 tok/s** aggregate (sweeps 57.1 / 49.9 / 58.2), mean time to first token 3.9 to 4.8 s, draft acceptance 0.40 to 0.41 |
| Single request, long reply | one request writing a 16,000-token reply, thinking at max, temperature 1.0 | about 30.5 tok/s (522 and 525 s per reply) |
| Long-reply quality | `tools/long_reply_check`: decode-time logprob of every token against a prefill rescoring of the same token ids, per 2,000-token window | 2 of 2 replies of 16,000 tokens clean: worst window mean \|diff\| 0.094, 5 tokens per 1,000 off by more than 1 nat |
| Corrupted output | `tools/corruption_probe.py`: 12 replies in Korean, Chinese, Japanese, Russian, Arabic and emoji-heavy English, greedy and temperature 1.0 | 0 U+FFFD, 0 surrogates, 0 control characters in 18,017 characters |
| Long prompt | one 94,933-token prompt through the sparse indexer's prefill path, greedy | 38.8 s, correct answer |
| GPU memory per rank | `nvidia-smi` process memory after boot, idle | 95,160 MiB; head node available memory 10,786 MiB |
| Boot | four ranks, weights on local NVMe | serving in about 5.5 minutes |

## Key findings

### 1. NVIDIA's checkpoint against a compressed-tensors NVFP4 checkpoint, same stack

We served both on the same build, launcher and load on the same day (only the weights and `--linear-backend marlin` for NVIDIA's NVFP4 dense layers differ). The other checkpoint was [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) at `18d55bf`.

| | NVIDIA | Red Hat |
|---|---|---|
| 8 sessions, aggregate tok/s (3 sweeps, hit rate 0.84) | 54.9 | 52.8 |
| 8 sessions, mean time to first token | 3.9 to 4.8 s | 3.8 to 4.7 s |
| DFlash2 draft acceptance | 0.40 to 0.41 | 0.32 to 0.37 |
| 16,000-token reply, one request | about 30.5 tok/s | 26 to 29 tok/s |
| Long-reply check, worst window | 0.094 | 0.091 |
| Corruption probe | 0 of 18,017 characters | 0 of 18,182 |
| GPU memory per rank | 95,160 MiB | 95,237 MiB |

The throughput lead is inside the run-to-run spread of one 3-sweep run each. The higher draft acceptance is consistent across every sweep and is where the single-request speed comes from.

### 2. Check a ModelOpt checkpoint before you trust it: vLLM #54150

[vllm#54150](https://github.com/vllm-project/vllm/issues/54150) (open at the time of writing) reports corrupted tokens from ModelOpt NVFP4 checkpoints of this model. The root cause, found by [mechramc](https://github.com/mechramc) in that thread: vLLM's fused gate/up repack gives each expert one global scale, the gate's, so an expert whose gate and up scales differ gets its up half mis-scaled. Whether that happens is a property of the checkpoint. `tools/check_gate_up_scales.py` reads those scales straight from Hugging Face with range requests, no download:

| Checkpoint | Expert pairs sampled | Gate scale differs from up scale |
|---|---|---|
| nvidia/GLM-5.3-Flash-NVFP4 @ 09b04e5 | 336 (8 per MoE layer) | 0 |
| LibertAIDAI/GLM-5.3-Flash-NVFP4 @ caca4e6 | 43 (1 per MoE layer) | 28 (65%) |

NVIDIA's official checkpoint shares the scale, so the bug cannot trigger, and our corruption probe agrees (finding 1).

### 3. An 8-bit drafter silently corrupted long replies

Measured on this stack with the Red Hat checkpoint. Packing the DFlash2 drafter's weights to 8 bits (patch 10) looked fine on short evals, but long replies broke a few thousand tokens in: fused words, the reasoning section ending mid-sentence, then collapse. With the 16-bit drafter, the same test is clean; with the 8-bit drafter loaded but nothing verified (draft length 0), it is clean too. An instrumented run found 0 mismatches in 5,200 verify steps between drafted, fed and compared tokens, so the root cause is still open. Scope: every run used `VLLM_MARLIN_USE_ATOMIC_ADD=1`, which changes how the 8-bit drafter's Marlin kernels reduce, and we did not test it off. Patch 10 stays in the series, off by default; do not enable it for serving. `tools/long_reply_check` is the check that caught it; we run it every night.

### 4. 1.9 GB per node back from one upstream fix

Measured on this stack with the Red Hat checkpoint. The sparse indexer's prefill workspace was sized in tokens, although its cache holds one row per 4 tokens; vLLM reserves it at startup and locks it. Patch u55222 carries the workspace half of [vllm#55222](https://github.com/vllm-project/vllm/pull/55222) (open at the time of writing): 1,900 to 1,920 MiB less GPU memory per rank, with identical output on a 94,933-token prompt.

### 5. A warm prefix cache inflates throughput numbers

Right after the runs above, three more 8-session sweeps on the NVIDIA checkpoint read 63.1 / 78.9 / 86.0 tok/s at a 0.96 hit rate, only because the prompts were already cached. Compare throughput only between runs with similar hit rates; `agent_sessions_long.py` reports the hit rate for every level.

### 6. Send reasoning effort high, not max

The chat template knows two efforts, `low` and `high`. Any other value, and a request that sends none, renders as `Reasoning Effort: Max`. At max the model plans a whole multi-step agent task inside its thinking before its first tool call. Measured on this stack on 2026-09-28, one real agent request (a 34K-token agent harness prompt with 38 tools and a five-step coding task), replayed with streaming and a 12,000-token cap:

| `reasoning_effort` | Thinking before the first tool call | First tool call |
|---|---|---|
| max | over 12,000 tokens, hit the cap still planning | none after 256 s |
| high | 63 tokens | 16 s |

In a desktop agent client the same task at max ran 11 minutes with no visible action (the server generated 15,786 tokens in the last 6 of them alone), and the client reported a dropped stream while the server was still generating. High is our standard for every client. The numbers in Results were measured at max, the heaviest setting, and the measurement tools still send max on purpose.

## Open questions

- **Is the NVIDIA checkpoint really faster at 8 sessions?** One run of 3 sweeps each: 54.9 against 52.8 tok/s, inside the spread. The acceptance gain is consistent; the throughput gain needs more sweeps.
- **Native MTP does not load** from this checkpoint: its layer 45 is BF16 but not on the quantization ignore list. This build uses DFlash2 and does not need it.
- **Does patch u55222 cost throughput?** On the Red Hat checkpoint, one run after it read 0.949x at the same cache hit rate, inside the spread and with no mechanism we can find; a clean side-by-side has not been run.
- **Why does the 8-bit drafter corrupt the target's output?** See finding 3.
- **Upstream work we did not carry yet:** [vllm#58684](https://github.com/vllm-project/vllm/pull/58684) waits on [#58260](https://github.com/vllm-project/vllm/pull/58260), an open fix that rewrites the same function; [vllm#58762](https://github.com/vllm-project/vllm/pull/58762) is being replaced by the open draft [#58851](https://github.com/vllm-project/vllm/pull/58851).

## What's here

| Path | What it is |
|---|---|
| `image-main/` | `build.sh`, the patch series (`patches/vllm/series`, 19 patches), two FlashInfer GB10 fixes, `EXPECTED.sha256`, and an optional NCCL build with a RoCE deadlock fix (`nccl/`) |
| `launch/launch_node.sh` | our production serving flags for one rank; only machine-specific values are variables |
| `template/` | Z.ai's official GLM-5.3-Flash chat template plus one explicit thinking opt-out line |
| `docs/patches.md` | every patch: what it does and where it came from |
| `tools/check_gate_up_scales.py` | checks a ModelOpt NVFP4 checkpoint for the vLLM #54150 defect without downloading it |
| `tools/corruption_probe.py` | multilingual corrupted-token probe |
| `tools/long_reply_check/` | the long-reply corruption check |
| `tools/agent_sessions_long.py` | the 8-session agent load behind the throughput numbers (needs a vLLM v0.30.0 checkout for its text pool) |
| `tools/agent_sessions.py` | a shorter agent-session load at 1, 2, 4 and 8 sessions |
| `tools/ab_harness.py`, `tools/analyze_trace.py`, `tools/quality_eval.py` | drafter A/B harness, trace analysis, NLL and GSM8K quality eval |
| `tools/lab/` | kernel and NCCL microbenchmarks and an oomwrap supervisor |

## Quick start

**1. Build the image on each node.** You need the base image locally (about 20 GB):
```bash
docker pull vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423
cd image-main && ./build.sh      # builds glm53-flash-gb10:main-7f1a
```
The script copies the target files out of the base image, applies the series in order plus the FlashInfer patches with zero fuzz, and refuses to build unless every result matches `EXPECTED.sha256`. The build only copies files; no container runs. To include the NCCL fix, run `nccl/build_nccl.sh` first (about 12 minutes on a Spark) and then `NCCL_LIB=nccl/libnccl.so.2 ./build.sh`.

**2. Get the weights onto every node** (not redistributed here; follow each license):
- Checkpoint: [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) at revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` (190.4 GiB), in `$HF_ROOT/hub/nvidia-glm53-flash-nvfp4/`.
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at revision `7d74cdd881ed7e32c31175984a67823127b66cfe`, in `$HF_ROOT/hub/glm53-flash-dflash2/`. We chose it over the two later revisions (`dc77ff1`, `bf582e4`) in A/Bs on an earlier stack and have not re-run that comparison on this checkpoint.

Download once and copy node to node over the fast link rather than downloading four times. We downloaded on one node in 33 minutes and fanned out over RoCE in 32.

**3. Put the chat template next to the checkpoint** on every node:
```bash
cp template/chat_template_zai0907_optout.jinja "$HF_ROOT/hub/nvidia-glm53-flash-nvfp4/"
```
The launcher passes it with `--chat-template` and refuses to start without it. It is Z.ai's official 09-07 template plus one explicit opt-out line: every request reasons unless it sends `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, it renders byte for byte like the official template. Have your clients send `"reasoning_effort": "high"`: the template falls back to Max when a request sends nothing (finding 6).

**4. Launch, workers first, head last:**
```bash
export NODES="<ip0> <ip1> <ip2> <ip3>"   # rail IPs, rank 0 (head) first
./launch/launch_node.sh 3   # on node 3, then 2 and 1
./launch/launch_node.sh 0   # on the head node
```
Defaults assume the Spark's QSFP port shows up as `enp1s0f0np0` / `enP2p1s0f0np0` (RDMA devices `rocep1s0f0` / `roceP2p1s0f0`) and RoCE GID index 3; see the variables at the top of the script. Point `CACHE` at a fresh directory for each image build (it holds the JIT caches). We run every rank under [oomwrap](https://github.com/osolmaz/oomwrap) with `tools/lab/oomwrap_supervise.sh`.

**5. Verify with a real completion,** not just `/health`:
```bash
curl -s http://HEAD:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","max_tokens":16,"messages":[{"role":"user","content":"Say OK"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```
For quality, run `tools/corruption_probe.py` and `tools/long_reply_check` against your server before trusting long outputs.

**Memory:** each Spark runs earlyoom (SIGTERM at 4% free, about 4.9 GB). The 35 GiB KV cache keeps the head node clear of that line.

## Credits

- **NVIDIA**: the official GLM-5.3-Flash NVFP4 checkpoint served here (MIT, ModelOpt). **[Z.ai](https://huggingface.co/zai-org) ([@Zai_org](https://x.com/Zai_org))**: GLM-5.3-Flash (MIT) and the official chat template.
- **[Humming](https://github.com/vllm-project/humming)** ([Jinzhen Lin](https://github.com/jinzhen-lin), [Julian Huang](https://github.com/huangzhilin-hzl), [Misha Goin](https://github.com/mgoin) ([@mgoin_](https://x.com/mgoin_)) and the Humming contributors, Apache-2.0): the MoE kernels.
- **[vLLM](https://github.com/vllm-project/vllm) ([@vllm_project](https://x.com/vllm_project))** and **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** (Apache-2.0). vLLM contributors whose pull requests we carry: [Matt Mastracci](https://github.com/mmastrac) ([@mmastrac](https://x.com/mmastrac); #58704, #58454), [Juntian777](https://github.com/Juntian777) (#57632), [shiweijiezero](https://github.com/shiweijiezero) (#58834), [QHarshil](https://github.com/QHarshil) (#58021), [drakosha](https://github.com/drakosha) (#55222), [JaredforReal](https://github.com/JaredforReal) (#57477, now in vLLM main), and vLLM's own revert #58250 of [mgoin](https://github.com/mgoin)'s fused DFlash2 grouped convolution (#55960). [mechramc](https://github.com/mechramc): the root cause of vLLM #54150.
- **Tony ([@2WildTech](https://x.com/2WildTech), [tonyd2wild](https://github.com/tonyd2wild))**: the GB10 patch set that patches 01 to 08 are ported from, and the two FlashInfer FP8 MLA fixes.
- **Sufyan ([@sfxnz](https://x.com/sfxnz), [sfxnz](https://github.com/sfxnz))**: his public recipe for NVIDIA's checkpoint on DGX Spark, including Marlin for its NVFP4 dense layers on GB10.
- **[incoai](https://huggingface.co/incoai)**: the GLM-5.3-Flash DFlash2 drafter (CC BY-NC-ND 4.0), built on DFlash from [Z Lab](https://github.com/z-lab/dflash) ([@zhijianliu_](https://x.com/zhijianliu_)).
- **[Jacopo Nardiello](https://github.com/jnardiello) ([@jnardiello](https://x.com/jnardiello))**: the 8-bit drafter experiment that patch 10 ports and the 8-bit residue idea behind patch p2 (both off in this build).
- **NCCL:** [Stanislav Bardyuk](https://github.com/kodlan) wrote [NVIDIA/nccl#2393](https://github.com/NVIDIA/nccl/pull/2393), the fix for the RoCE deadlock reported in [#2334](https://github.com/NVIDIA/nccl/issues/2334); [Rami Nudelman](https://github.com/raminudelman) at NVIDIA brought it to us.
- **[oomwrap](https://github.com/osolmaz/oomwrap)** by [Onur Solmaz](https://github.com/osolmaz) ([@onusoz](https://x.com/onusoz)): memory-pressure supervision for every launch.

## License

The scripts, tools and patches in this repo are Apache-2.0 (see `LICENSE`); the patches modify Apache-2.0 vLLM and FlashInfer code and BSD-3-Clause NCCL code. Humming (Apache-2.0) is not included here; the build selects it with `--moe-backend humming`. `template/` is Z.ai's official chat template with one added line and keeps Z.ai's license. No model weights are included. Each model has its own license: NVIDIA's checkpoint is MIT, and the incoai drafter is non-commercial and no-derivatives.
