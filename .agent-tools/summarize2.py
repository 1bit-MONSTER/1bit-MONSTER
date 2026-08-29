import sys, json
def txt(c):
    if isinstance(c, list):
        return ' '.join(str(x.get('text','')) for x in c if isinstance(x,dict))
    return str(c)
users, assis, title = [], [], None
for l in sys.stdin:
    if not l.strip(): continue
    try: m = json.loads(l)
    except: continue
    t = m.get('type')
    if t == 'session/title':
        d = m.get('data', {})
        title = d.get('title') if isinstance(d, dict) else None
    elif t == 'user/message':
        d = m.get('data', {})
        users.append(txt(d.get('content','')))
    elif t == 'assistant/message':
        d = m.get('data', {})
        mm = d.get('message', {}) if isinstance(d, dict) else {}
        assis.append(txt(mm.get('content','')))
print('TITLE:', str(title)[:120])
if users:
    print('FIRST USER:', users[0][:200].replace('\n',' '))
    print('LAST USER :', users[-1][:200].replace('\n',' '))
if assis:
    print('FINAL ASSISTANT:', assis[-1][:800].replace('\n',' '))
