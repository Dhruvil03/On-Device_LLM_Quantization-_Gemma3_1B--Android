"""
Step 1: Inspect a converted .tflite model to see its actual signatures,
input names, and output names. Run this BEFORE the generation script,
since the exact tensor names depend on conversion flags
(--mask_as_input, --transpose_kv_cache, --kv_cache_max_len, etc.)
and can vary between quant levels / conversion runs.
"""

import sys

# Try the LiteRT-native interpreter first, fall back to TF's if needed
try:
    from ai_edge_litert.interpreter import Interpreter
    print("Using ai_edge_litert.interpreter.Interpreter")
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter (fallback)")
    except ImportError:
        print("Neither ai_edge_litert nor tensorflow found.")
        print("Install with: pip install ai-edge-litert")
        sys.exit(1)

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "gemma3-1b_int4_q4_block128_ekv1280.tflite"

print(f"\nLoading: {MODEL_PATH}\n")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

sig_list = interpreter.get_signature_list()
print(f"Signatures found: {list(sig_list.keys())}\n")

for sig_name, sig_info in sig_list.items():
    print(f"=== Signature: {sig_name} ===")
    print(f"  Inputs:  {sig_info['inputs']}")
    print(f"  Outputs: {sig_info['outputs']}")

    # Get the actual runner to inspect tensor shapes/dtypes
    runner = interpreter.get_signature_runner(sig_name)
    print("  Input details:")
    for name, detail in runner.get_input_details().items():
        print(f"    {name}: shape={detail['shape']}, dtype={detail['dtype']}")
    print("  Output details:")
    for name, detail in runner.get_output_details().items():
        print(f"    {name}: shape={detail['shape']}, dtype={detail['dtype']}")
    print()