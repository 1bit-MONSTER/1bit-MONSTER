import sys, json
msgs = [json.loads(l) for l in sys.stdin if l.strip()]
print('total lines:', len(msgs))
texts = [m for m in msgs if m.get('type') == 'message']
print('message lines:', len(texts))
for m in texts[-8:]:
    c = m.get('content', '')
    if isinstance(c, list):
        c = ' '.join(str(x.get('text', '')) for x in c if isinstance(x, dict))
    s = str(c).replace('\n', ' ')
    print(m.get('role', '?'), '->', s[:400])
