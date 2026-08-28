import sys, json, collections
types = collections.Counter()
for l in sys.stdin:
    if not l.strip(): continue
    try: m = json.loads(l)
    except: continue
    types[m.get('type', '?')] += 1
print('types:', dict(types))
