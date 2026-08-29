import sys, json
def txt(c):
    if isinstance(c, list):
        return ' '.join(str(x.get('text', '')) for x in c if isinstance(x, dict))
    return str(c)
users, assis, tools, title = [], [], [], None
for l in sys.stdin:
    if not l.strip(): continue
    try: m = json.loads(l)
    except: continue
    t = m.get('type')
    if t == 'session/title': title = m.get('data') or m.get('title')
    elif t == 'user/message': users.append(txt(m.get('content', '')))
    elif t == 'assistant/message': assis.append(txt(m.get('content', '')))
    elif t == 'tool/call': tools.append(m.get('name', '?'))
print('TITLE:', title)
if users:
    print('FIRST USER:', users[0][:300].replace('\n', ' '))
    print('LAST USER :', users[-1][:300].replace('\n', ' '))
if assis:
    print('FINAL ASSISTANT:', assis[-1][:600].replace('\n', ' '))
if tools:
    from collections import Counter
    print('TOOLS:', dict(Counter(tools)))
