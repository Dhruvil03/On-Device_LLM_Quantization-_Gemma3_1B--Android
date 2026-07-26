# Gemma 3 1B On-Device Quantization: Final Results

**Device:** Moto G Stylus 5G (2023), 5.5GB RAM, ~2.3GB typically available
**Model:** google/gemma-3-1b-it, converted via LiteRT-Torch to TFLite
**Benchmark tool:** LiteRT `benchmark_model` (CLI), 5 runs per config, CPU-only, XNNPACK delegate

## Summary Table

| Metric | FP16 (baseline, not deployed) | INT8 (lite, prefill_128) | INT4 (block128, prefill_8) |
|---|---|---|---|
| File size | ~2 GB (est.) | 1020.4 MB | 527.2 MB |
| Init time (median) | -- | 4637.4 ms | 4568.2 ms |
| **Inference avg (median)** | -- | **1083.4 ms** | **89.4 ms** |
| Inference stdev (5 runs) | -- | 14.8 ms (stable) | 40.8 ms (variable) |
| Overall memory footprint (median) | -- | 2733.8 MB | 2907.2 MB |
| **Runs on full 7-signature build?** | N/A | **No — hung/OOM** | Yes |
| **Runs on 2-signature (lite) build?** | N/A | Yes (barely fits) | Yes |
| Accuracy vs fp16 (15 prompts) | ground truth | 10 Match / 3 Degraded / **2 Broken** | 8 Match / 5 Degraded / **2 Wrong** |

**Important caveat on the latency comparison:** INT8 was benchmarked on the `prefill_128` signature (128 tokens) while INT4 was benchmarked on `prefill_8` (8 tokens) — a 16x difference in sequence length. Part of the ~12x latency gap reflects this signature mismatch, not quantization alone. A same-signature comparison was not obtained due to memory constraints preventing a full 7-signature INT8 build from running.

## Key Findings

**1. Memory is the binding constraint, not raw quantization level.**
The full INT8 export (7 signatures: prefill_8/64/128/256/512/1024 + decode, ~1076 subgraphs) consistently hung the device during initialization — likely exceeding available RAM. A reduced INT8 export (2 signatures: prefill_128 + decode) barely fit, running at a median 2734MB footprint against ~2.3GB typically available. INT4 ran reliably across all configurations.

**2. INT4 and INT8 fail in qualitatively different ways.**
- INT8's rare failures (2/15 prompts) were **catastrophic**: complete degeneration into repeated non-English script tokens (Tamil, Persian).
- INT4's more frequent failures (7/15 prompts) were **milder**: repetition loops (same sentence repeated) and occasional factual hallucination, but always stayed coherent in English.
- Neither quant level was uniformly "worse" — on 2/15 prompts, INT4 actually completed answers more fully than the fp16 baseline (which was cut off by the 60-token budget).

**3. INT4 latency was reproducible but noisy; INT8 was slow but stable.**
INT4's inference time varied nearly 2.4x across 5 runs (69-169ms), likely due to its smaller memory footprint leaving more room for system-level interference. INT8's tighter memory margin appears to have forced more consistent (if much slower) scheduling behavior (15ms stdev vs 41ms for INT4).

## What Would Strengthen This Further
- A true same-signature latency comparison (e.g. both INT4 and INT8 on prefill_8)
- A larger accuracy eval set (15 prompts is a reasonable pilot, not a rigorous benchmark)
- GPU/NNAPI delegate benchmarks, not just CPU
- Testing on a second device to see if the memory-constraint finding generalizes or is specific to this phone's ~5.5GB RAM tier
