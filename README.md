# On-Device LLM Quantization: Gemma 3 1B on Android

Quantizing and deploying a small LLM (Gemma 3 1B) on-device, benchmarking real
latency/memory/accuracy trade-offs on actual Android hardware rather than
relying on synthetic or emulator numbers.

**TL;DR:** INT4 quantization ran reliably and fast (~89ms median inference);
full INT8 exceeded the test device's memory headroom and hung; a reduced
INT8 build ran but ~12x slower. Accuracy degraded differently at each quant
level — INT8 rarely failed but catastrophically (garbled non-English output);
INT4 failed more often but mildly (repetition loops, staying coherent).

<img width="2580" height="1111" alt="pareto_chart" src="https://github.com/user-attachments/assets/1fc8f300-cb76-43df-92da-582df705da67" />


---

## Motivation

Most quantization write-ups report a clean accuracy-vs-compression curve
measured on desktop/server GPUs. This project instead asks the more practical
question a mobile ML engineer actually faces: **what happens when you deploy
on an actual mid-range phone with limited RAM?** The answer here — that
memory pressure, not just arithmetic precision, is the dominant failure mode
— isn't obvious from benchmarking on a workstation.

---

## Setup

| | |
|---|---|
| **Model** | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) |
| **Conversion toolchain** | [LiteRT-Torch](https://github.com/google-ai-edge/litert-torch) (Google AI Edge) |
| **Target device** | Moto G Stylus 5G (2023) — 5.5 GB RAM, ~2.3 GB typically available |
| **Quantization levels tested** | `dynamic_int8`, `dynamic_int4_block128` |
| **Benchmark tool** | LiteRT `benchmark_model` CLI (prebuilt Android ARM64 binary) |
| **Inference backend** | XNNPACK delegate, CPU, 2-4 threads |

---

## Pipeline

```
HuggingFace checkpoint (fp16)
        │
        ▼
LiteRT-Torch conversion  ──►  .tflite (multi-signature: prefill_N × 6, decode)
        │                          │
        ▼                          ▼
   Quantize (INT8 / INT4)     Push to phone via adb
        │                          │
        ▼                          ▼
  benchmark_model (CLI)      Real text generation
  latency / memory / size    (prefill + decode loop,
                               manual KV cache mgmt)
        │                          │
        └──────────┬───────────────┘
                    ▼
         Compare vs fp16 baseline
         (accuracy + speed + memory)
```

1. **Convert**: `convert_gemma3_to_tflite.py --model_size=1b --quantize=<level>`
2. **Benchmark**: push `.tflite` + `benchmark_model` binary to `/data/local/tmp/`, run via `adb shell`
3. **Generate**: load the `.tflite` graph with the LiteRT Python interpreter, manually drive the `prefill_*` and `decode` signatures (no built-in `.generate()` for raw TFLite graphs)
4. **Compare**: score INT4/INT8 outputs against an fp16 HuggingFace `transformers` baseline on the same 15 prompts

---

## Results

### Latency, memory, size

| Metric | INT8 (lite, `prefill_128`) | INT4 (`block128`, `prefill_8`) |
|---|---|---|
| File size | 1020.4 MB | 527.2 MB |
| Init time (median, 5 runs) | 4637 ms | 4568 ms |
| **Inference avg (median, 5 runs)** | **1083.4 ms** | **89.4 ms** |
| Inference stdev | 14.8 ms (stable) | 40.8 ms (noisy) |
| Overall memory footprint (median) | 2733.8 MB | 2907.2 MB |
| Runs on full 7-signature build? | ❌ Hung / OOM | ✅ Yes |
| Runs on 2-signature (lite) build? | ✅ Yes, barely fits | ✅ Yes |

> **Caveat:** INT8 was benchmarked on `prefill_128` (128 tokens), INT4 on
> `prefill_8` (8 tokens) — a 16x sequence-length difference. Part of the ~12x
> latency gap reflects this, not quantization precision alone. A true
> same-signature comparison wasn't obtained because the full multi-signature
> INT8 build didn't fit in memory. See [Limitations](#limitations).

### Accuracy (15 prompts vs. fp16 baseline)

| | Match | Degraded | Wrong | Broken |
|---|---|---|---|---|
| **INT8** | 10/15 | 3/15 | 0/15 | **2/15** |
| **INT4** | 8/15 | 5/15 | **2/15** | 0/15 |

- **Match**: same answer/meaning as fp16
- **Degraded**: coherent but worse (verbose, repeats itself, minor artifacts)
- **Wrong**: factually incorrect or logically confused
- **Broken**: generation collapsed into garbled/non-English tokens

Full per-prompt breakdown: [`merged_comparison.json`](merged_comparison.json)

---

## Key Findings

### 1. Memory is the binding constraint — not quantization level by itself
The full INT8 export (7 signatures: `prefill_8/64/128/256/512/1024` + `decode`,
~1076 subgraphs after MLIR lowering) consistently hung the device during
model initialization. A reduced 2-signature INT8 export ran, but at a median
2734 MB footprint against a device with only ~2.3 GB typically available —
right at the edge. INT4 ran comfortably across every configuration tested.
**Takeaway:** on memory-constrained hardware, whether a quantization level is
viable can matter more than how fast it runs once it *does* fit.

### 2. INT4 and INT8 fail in qualitatively different ways
- **INT8's failures were rare but catastrophic**: 2/15 prompts degenerated
  into repeated tokens in unrelated scripts (Tamil, Persian) — a complete
  generation collapse, not just lower quality.
- **INT4's failures were common but mild**: 7/15 prompts showed repetition
  loops or occasional factual hallucination, but generation never left
  coherent English.
- Neither level was uniformly worse: on 2/15 prompts INT4 completed answers
  more fully than the fp16 baseline (which was truncated by the token budget).

### 3. Run-to-run latency variance differed sharply between quant levels
INT4 inference time varied ~2.4x across 5 runs (69–169 ms); INT8 was far more
stable (1068–1105 ms). Hypothesis: INT4's smaller memory footprint leaves more
slack for other system activity (thermal state, background processes) to
perturb timing, while INT8's tighter memory margin forces more consistent —
if much slower — scheduling behavior. Not confirmed, just an observation
worth a follow-up.

---

## Limitations & What Would Strengthen This

- **No same-signature latency comparison.** The single strongest
  methodological gap — INT4 vs INT8 timings aren't directly comparable due to
  differing sequence lengths benchmarked.
- **Single device.** Results (especially the memory-hang finding) may be
  specific to this phone's ~5.5 GB RAM tier; untested on higher/lower-RAM
  devices.
- **15-prompt eval is a pilot, not a rigorous benchmark.** Enough to show a
  real qualitative pattern, not enough for statistical confidence.
- **CPU-only.** No GPU/NNAPI delegate benchmarks were run; on-device NPU
  acceleration could change the latency picture substantially.
- **`decode` signature not separately benchmarked.** All latency numbers here
  are prefill (prompt-processing); `decode` (per-token generation speed) is
  arguably more representative of real chat-style usage and wasn't isolated.

---

## Repo Structure

```
.
├── README.md                        # this file
├── convert_gemma3_to_tflite.py      # LiteRT-Torch conversion (from google-ai-edge/litert-torch)
├── inspect_tflite_model.py          # dumps signature/tensor info from a .tflite file
├── run_fp16_baseline.py             # fp16 HF baseline generation (ground truth)
├── run_tflite_generation.py         # manual prefill+decode generation loop for .tflite models
├── benchmark_summary.csv            # INT4 latency/memory results, 5 runs
├── benchmark_summary_int8.csv       # INT8 latency/memory results, 5 runs
├── fp16_baseline_results.json       # fp16 outputs for 15 eval prompts
├── int4_results.json                # INT4 outputs for the same prompts
├── int8_results.json                # INT8 outputs for the same prompts
├── merged_comparison.json           # side-by-side comparison + quality labels
├── pareto_chart.py                  # generates pareto_chart.png
├── pareto_chart.png                 # final trade-off visualization
└── final_results_summary.md         # condensed findings (superseded by this README)
```

---

## Reproducing This

```bash
# 1. Convert (choose --quantize=dynamic_int8 | dynamic_int4_block128 | dynamic_int4_block32 | none)
python convert_gemma3_to_tflite.py \
  --model_size=1b \
  --checkpoint_path=./gemma3-1b-it \
  --output_path=./output/ \
  --quantize=dynamic_int4_block128

# 2. Benchmark on-device
adb push output/*.tflite /data/local/tmp/
adb push benchmark_model /data/local/tmp/
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/<model>.tflite \
  --num_threads=4 --warmup_runs=5 --num_runs=20 --min_secs=0

# 3. Generate text for accuracy comparison
python inspect_tflite_model.py output/<model>.tflite      # confirm tensor names first
python run_tflite_generation.py output/<model>.tflite results.json
```

---

## Acknowledgments

Built using Google's [LiteRT](https://developers.google.com/edge/litert) /
[LiteRT-Torch](https://github.com/google-ai-edge/litert-torch) toolchain and
the [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it)
checkpoint.
