import sys, json
for l in sys.stdin:
    if not l.strip(): continue
    try: m = json.loads(l)
    except: continue
    t = m.get('type')
    if t in ('user/message', 'assistant/message', 'session/title'):
        print('TYPE:', t)
        print('KEYS:', list(m.keys()))
        print('DATA:', json.dumps(m.get('data', m))[:700])
        print('---')
