import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("/home/bcloud/zaya1-8b-hf")
base = AutoModelForCausalLM.from_pretrained("/home/bcloud/zaya1-8b-hf", torch_dtype=torch.bfloat16)
ft = PeftModel.from_pretrained(base, "/home/bcloud/zaya-ft-lora")
prompt = tok.apply_chat_template([{"role": "user", "content": "Write a Python function named is_prime."}], tokenize=False)
ids = tok(prompt, return_tensors="pt")
for name, m in [("BASE", base), ("LORA-FT", ft)]:
    m.eval()
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"\n=== {name} ===\n" + tok.decode(out[0][ids['input_ids'].shape[1]:])[:400])
