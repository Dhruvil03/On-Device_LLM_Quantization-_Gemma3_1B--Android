from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

prompts = [
    # Factual
    "What's the capital of France?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the chemical symbol for gold?",
    "In what year did World War II end?",

    # Reasoning / math
    "If a train leaves at 3pm going 60mph and travels for 2.5 hours, how far does it go?",
    "A store has 120 apples. They sell 45 in the morning and 30 in the afternoon. How many are left?",
    "If all cats are mammals, and all mammals are animals, are all cats animals? Explain briefly.",
    "What is 17 multiplied by 6?",

    # Open-ended / creative
    "Write two sentences about autumn.",
    "Describe a good cup of coffee in one sentence.",
    "Give me three tips for staying focused while studying.",

    # Instruction following
    "List the primary colors.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Explain what photosynthesis is in simple terms.",

    # Longer-form coherence test
    "Write a short paragraph (3-4 sentences) about why exercise is important.",
]

results = []
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"PROMPT: {p}")
    print(f"OUTPUT: {text}")
    print("-" * 60)
    results.append({"prompt": p, "fp16_output": text})

with open("fp16_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to fp16_baseline_results.json")