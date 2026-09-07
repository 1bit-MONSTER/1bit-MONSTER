import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("/home/bcloud/zaya1-8b-hf")
target_code = "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n"
for name, load in [("BASE", lambda: AutoModelForCausalLM.from_pretrained("/home/bcloud/zaya1-8b-hf", torch_dtype=torch.bfloat16)),
                   ("LORA-FT", lambda: PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("/home/bcloud/zaya1-8b-hf", torch_dtype=torch.bfloat16), "/home/bcloud/zaya-ft-lora"))]:
    m = load(); m.eval()
    msgs = [{"role":"user","content":"Write a Python function named is_prime."},{"role":"assistant","content":target_code}]
    full = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(full, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = m(ids, labels=ids)
    print(f"{name}: loss {out.loss.item():.4f} -> ppl {out.loss.item():.2e}")
