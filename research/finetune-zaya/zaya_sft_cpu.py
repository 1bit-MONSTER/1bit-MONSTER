import torch, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# tiny code-domain set (for a REAL adapter artifact)
progs = {
 "fibonacci": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
 "factorial": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
 "is_prime": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
 "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
 "fizzbuzz": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n",
 "sum_list": "def sum_list(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
 "binary_search": "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        elif xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
 "palindrome": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n",
}
data = []
for name, code in progs.items():
    q = f"Write a Python function named {name}."
    data.append({"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": code}]})

from datasets import Dataset
tok = AutoTokenizer.from_pretrained("/home/bcloud/zaya1-8b-hf")
model = AutoModelForCausalLM.from_pretrained("/home/bcloud/zaya1-8b-hf", torch_dtype=torch.bfloat16)
print("model loaded:", sum(p.numel() for p in model.parameters())/1e9, "B")
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=["o_proj","q_proj","k_proj","v_proj_current"], task_type="CAUSAL_LM")
pm = get_peft_model(model, cfg)
print("trainable:", sum(p.numel() for p in pm.parameters() if p.requires_grad))
scfg = SFTConfig(
    output_dir="/home/bcloud/zaya-ft-out", max_steps=12, per_device_train_batch_size=2,
    learning_rate=2e-4, logging_steps=2, save_strategy="no", report_to=[],
    max_length=256, gradient_accumulation_steps=1, bf16=False, fp16=False,
    optim="adamw_torch", lr_scheduler_type="cosine", seed=7, router_aux_loss_coef=0.0)
ds = Dataset.from_list([{"text": tok.apply_chat_template(m["messages"], tokenize=False)} for m in data])
trainer = SFTTrainer(model=pm, args=scfg, train_dataset=ds, processing_class=tok)
t0 = time.time()
trainer.train()
print("train done in %.0fs" % (time.time()-t0))
pm.save_pretrained("/home/bcloud/zaya-ft-lora")
print("adapter saved to ~/zaya-ft-lora")
