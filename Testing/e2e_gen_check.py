import torch, subprocess, sys
from transformers import AutoModelForCausalLM, AutoConfig
from llama_cpp import Llama
import numpy as np

torch.set_grad_enabled(False)
dir, N = sys.argv[1], 20
cfg = AutoConfig.from_pretrained(dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(dir, torch_dtype=torch.float32, config=cfg)
model.eval()
llm = Llama(model_path=f"{dir}/oracle-q8.gguf", n_ctx=512, n_gpu_layers=0, verbose=False)
p = "The capital of France is"
ids = llm.tokenize(p.encode(), add_bos=False)
with open("/tmp/ids.txt", "w") as f: f.write(" ".join(map(str, ids)))
st = subprocess.run(["/tmp/e2e_seq", dir, "/tmp/ids.txt", str(N)], capture_output=True, text=True, timeout=600)
seg = st.stdout.split("engine-gen:")[1].split()
eng = [int(x) for x in seg if x.lstrip('-').isdigit()]
with torch.no_grad():
    input_ids = torch.tensor([ids])
    torch_gen = []
    for i in range(N):
        nxt = int(model(input_ids).logits[0, -1].argmax())
        torch_gen.append(nxt)
        input_ids = torch.cat([input_ids, torch.tensor([[nxt]])], dim=1)
        if len(input_ids[0]) > 100: input_ids = input_ids[:, -100:]
m = sum(1 for a, b in zip(eng, torch_gen) if a == b)
print(f"{dir.split('/')[-1]}: {m}/{N} tokens identical")
print("  engine:", repr(llm.detokenize(eng)))
print("  torch :", repr(llm.detokenize(torch_gen)))
