"""
Step 2: Generate text from a converted .tflite Gemma3 model using its
prefill_* and decode signatures, for the same prompts used in the
fp16 baseline (run_fp16_baseline.py), so outputs can be compared directly.

Usage:
    python run_tflite_generation.py <path_to_tflite_model> <output_json_name>

Example:
    python run_tflite_generation.py output/gemma3-1b_int4_q4_block128_ekv1280.tflite int4_results.json
"""

import sys
import json
import numpy as np
from transformers import AutoTokenizer

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = sys.argv[1]
OUTPUT_JSON = sys.argv[2] if len(sys.argv) > 2 else "tflite_results.json"
MAX_NEW_TOKENS = 60  # matches the fp16 baseline
KV_CACHE_MAX_LEN = 1280  # matches --kv_cache_max_len used at conversion
NUM_LAYERS = 26  # kv_cache_k_0 .. kv_cache_k_25

# Same 15 prompts as run_fp16_baseline.py
PROMPTS = [
    "What's the capital of France?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the chemical symbol for gold?",
    "In what year did World War II end?",
    "If a train leaves at 3pm going 60mph and travels for 2.5 hours, how far does it go?",
    "A store has 120 apples. They sell 45 in the morning and 30 in the afternoon. How many are left?",
    "If all cats are mammals, and all mammals are animals, are all cats animals? Explain briefly.",
    "What is 17 multiplied by 6?",
    "Write two sentences about autumn.",
    "Describe a good cup of coffee in one sentence.",
    "Give me three tips for staying focused while studying.",
    "List the primary colors.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Explain what photosynthesis is in simple terms.",
    "Write a short paragraph (3-4 sentences) about why exercise is important.",
]

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

print(f"Loading model: {MODEL_PATH}")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

sig_list = interpreter.get_signature_list()
prefill_names = sorted(
    [s for s in sig_list if s.startswith("prefill_")],
    key=lambda s: int(s.split("_")[1])
)
decode_runner = interpreter.get_signature_runner("decode")


def make_empty_kv_cache():
    """Create zeroed KV cache tensors matching the model's expected shapes."""
    cache = {}
    for i in range(NUM_LAYERS):
        cache[f"kv_cache_k_{i}"] = np.zeros((1, 1, KV_CACHE_MAX_LEN, 256), dtype=np.float32)
        cache[f"kv_cache_v_{i}"] = np.zeros((1, 1, 256, KV_CACHE_MAX_LEN), dtype=np.float32)
    return cache


def pick_prefill_signature(n_tokens):
    """Pick the smallest prefill_N signature that fits n_tokens, padding as needed."""
    for name in prefill_names:
        n = int(name.split("_")[1])
        if n >= n_tokens:
            return name, n
    # fallback to the largest available
    largest = prefill_names[-1]
    return largest, int(largest.split("_")[1])


def build_mask(seq_len, total_len, kv_len=KV_CACHE_MAX_LEN):
    """Causal mask: 0 where attention allowed, large negative where masked."""
    mask = np.full((1, 1, seq_len, kv_len), -1e9, dtype=np.float32)
    for i in range(seq_len):
        allowed_upto = total_len - seq_len + i + 1
        mask[0, 0, i, :allowed_upto] = 0.0
    return mask


def generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0].astype(np.int32)
    n_prompt = len(input_ids)

    kv_cache = make_empty_kv_cache()

    # --- Prefill: process the prompt tokens (pad to the chosen signature length) ---
    sig_name, sig_len = pick_prefill_signature(n_prompt)
    prefill_runner = interpreter.get_signature_runner(sig_name)

    padded_tokens = np.zeros((1, sig_len), dtype=np.int32)
    padded_tokens[0, :n_prompt] = input_ids
    input_pos = np.arange(sig_len, dtype=np.int32)
    mask = build_mask(sig_len, sig_len)

    prefill_inputs = {"tokens": padded_tokens, "input_pos": input_pos, "mask": mask}
    prefill_inputs.update(kv_cache)

    prefill_out = prefill_runner(**prefill_inputs)
    for i in range(NUM_LAYERS):
        kv_cache[f"kv_cache_k_{i}"] = prefill_out[f"kv_cache_k_{i}"]
        kv_cache[f"kv_cache_v_{i}"] = prefill_out[f"kv_cache_v_{i}"]

    # --- Decode loop ---
    generated_ids = []
    cur_pos = n_prompt  # first decode step predicts the token after the prompt
    # Get the last real prompt token to feed as the first decode input
    next_token = int(input_ids[-1])

    for step in range(max_new_tokens):
        tokens_in = np.array([[next_token]], dtype=np.int32)
        input_pos_in = np.array([cur_pos - 1], dtype=np.int32)
        mask_in = np.full((1, 1, 1, KV_CACHE_MAX_LEN), -1e9, dtype=np.float32)
        mask_in[0, 0, 0, :cur_pos] = 0.0

        decode_inputs = {"tokens": tokens_in, "input_pos": input_pos_in, "mask": mask_in}
        decode_inputs.update(kv_cache)

        decode_out = decode_runner(**decode_inputs)
        for i in range(NUM_LAYERS):
            kv_cache[f"kv_cache_k_{i}"] = decode_out[f"kv_cache_k_{i}"]
            kv_cache[f"kv_cache_v_{i}"] = decode_out[f"kv_cache_v_{i}"]

        logits = decode_out["logits"][0, 0]
        next_token = int(np.argmax(logits))
        generated_ids.append(next_token)
        cur_pos += 1

        if next_token == tokenizer.eos_token_id:
            break

    full_ids = list(input_ids) + generated_ids
    return tokenizer.decode(full_ids, skip_special_tokens=True)


results = []
for p in PROMPTS:
    print(f"\nPROMPT: {p}")
    try:
        out = generate(p)
        print(f"OUTPUT: {out}")
        results.append({"prompt": p, "output": out})
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"prompt": p, "output": None, "error": str(e)})

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {OUTPUT_JSON}")