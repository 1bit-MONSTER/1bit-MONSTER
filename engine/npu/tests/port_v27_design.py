#!/usr/bin/env python3
# port_v27_design.py — convert modern-DSL aie.dma_bd operand-list form
# (buf, OFF, LEN, [<size=N,stride=M>...]) to the attr form the
# iron-lane pip aiecc parses (offset = N len = M sizes=[..] strides=[..]).
# modern: aie.dma_bd(%buf : memref<...>, OFF, LEN, [<size = s, stride = t>, ...]) {burst_length = B : i32}
# target: aie.dma_bd(%buf : memref<...> offset = OFF len = LEN sizes = [s..] strides = [t..]) {burst_length = B : i32}
pat = re.compile(
    r'aie\.dma_bd\((%[A-Za-z0-9_]+ : memref<[^>]*>),\s*(\d+),\s*(\d+),\s*\['
    r'(<size = (\d+), stride = (\d+)>(?:,\s*<size = (\d+), stride = (\d+)>)?(?:,\s*<size = (\d+), stride = (\d+)>)?(?:,\s*<size = (\d+), stride = (\d+)>)?)\]\s*\)')

count = [0]
def repl(m):
    count[0] += 1
    buf, off, ln = m.group(1), m.group(2), m.group(3)
    pairs = [(int(m.group(i)), int(m.group(i+1))) for i in range(5, 13, 2) if m.group(i) is not None]
    # modern lists dims outermost-first? iron example lists sizes=[1,1,1,256] strides=[0,0,0,1]
    # for a K-dim transfer with the innermost = (256,1) last in iron; modern emits
    # [<size,stride>...] in the same order as the generator's dims argument.
    sizes = ",".join(str(s) for s, _ in pairs)
    strides = ",".join(str(t) for _, t in pairs)
    return f"aie.dma_bd({buf} offset = {off} len = {ln} sizes = [{sizes}] strides = [{strides}])"

out = pat.sub(repl, src)
print(f"converted {count[0]} dma_bd ops", file=sys.stderr)
open(sys.argv[2], "w").write(out)
