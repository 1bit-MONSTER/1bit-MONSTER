#!/usr/bin/env python3
"""convert_model.py — one command: model source → INT8 ONNX + .htok + config.

usage: convert_model.py <name> <source> [--tokenizer tokenizer.json] [--out DIR]

  source: model.safetensors (HF bf16)  → safetensors converter (+ HF tokenizer.json for .htok)
          model.gguf (F32/F16/Q8_0)    → gguf converter (htok built from the gguf)

Outputs <out>/<name>/model_int8.onnx + config.json + <name>.htok.
Serve with: bitnet_decode <out>/<name>/model_int8.onnx --server 8080 --tokenizer <out>/<name>/<name>.htok --config <out>/<name>/config.json
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('source')
    ap.add_argument('--tokenizer', default=None, help='HF tokenizer.json (safetensors route)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'int8'))
    args = ap.parse_args()

    src = os.path.abspath(args.source)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    tmp = out + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    py = sys.executable
    if src.endswith('.safetensors') or os.path.isdir(src) or src.endswith('index.json'):
        subprocess.run([py, os.path.join(HERE, 'safetensors_to_onnx_int8.py'), src, tmp], check=True)
        if args.tokenizer:
            subprocess.run([py, os.path.join(HERE, 'tokenizer_json_to_htok.py'), args.tokenizer,
                            os.path.join(tmp, args.name + '.htok')], check=True)
        else:
            print('[convert] WARN: no --tokenizer — skipping .htok (serve needs it)')
    elif src.endswith('.gguf'):
        subprocess.run([py, os.path.join(HERE, 'gguf_to_onnx_int8.py'), src, tmp], check=True)
        htok = os.path.join(ROOT, 'build', 'gguf_htok')
        if os.path.exists(htok):
            subprocess.run([htok, src, os.path.join(tmp, args.name + '.htok')], check=True)
        else:
            print('[convert] WARN: build/gguf_htok missing — skipping .htok')
    else:
        sys.exit(f'unknown source type: {src} (want .safetensors or .gguf)')

    for f in os.listdir(tmp):
        shutil.move(os.path.join(tmp, f), os.path.join(out, f))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'[convert] done → {out}')
    print(f'[convert] serve: bitnet_decode {out}/model_int8.onnx --server 8080 '
          f'--tokenizer {out}/{args.name}.htok --config {out}/config.json')

if __name__ == '__main__':
    main()
