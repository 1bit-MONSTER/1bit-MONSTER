# Oracle = llama.cpp (pilot #9 fallback for archs torch 5.x dropped).
# Engine (f32 safetensors) vs llama.cpp (Q8 GGUF): Q8 quantization means
# near-tie flips are EXPECTED — 18-20/20 with coherent text = validated.
import subprocess, sys
from llama_cpp import Llama
dir, N = sys.argv[1], 20
llm = Llama(model_path=f"{dir}/oracle-q8.gguf", n_ctx=512, n_gpu_layers=0, verbose=False)
p = "The capital of France is"
ids = llm.tokenize(p.encode(), add_bos=False)
with open("/tmp/ids.txt", "w") as f: f.write(" ".join(map(str, ids)))
st = subprocess.run(["/tmp/e2e_seq", dir, "/tmp/ids.txt", str(N)], capture_output=True, text=True, timeout=600)
seg = st.stdout.split("engine-gen:")[1].split()
eng = [int(x) for x in seg if x.lstrip('-').isdigit()]
# llama.cpp greedy reference: n_predict=N, deterministic (temp 0)
out = llm.create_completion(p, max_tokens=N, temperature=0.0, top_k=1, echo=False)
lc = llm.tokenize(out["choices"][0]["text"].encode(), add_bos=False)[:N]
m = sum(1 for a, b in zip(eng, lc) if a == b)
print(f"{dir.split('/')[-1]}: {m}/{N} tokens identical (engine f32 vs llama.cpp Q8)")
print("  engine:", repr(llm.detokenize(eng)))
print("  llama :", repr(llm.detokenize(lc)))
