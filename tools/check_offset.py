"""Check where weight data actually starts in the 1BP file"""
import struct

with open('/home/bcloud/1bit-monster/models/ZAYA1-8B.1bp', 'rb') as f:
    f.seek(256)
    for i in range(400):
        nl = struct.unpack('<I', f.read(4))[0]
        if nl > 120: break
        name = f.read(nl).decode()
        null = f.read(1)
        ndim = struct.unpack('<I', f.read(4))[0]
        d0, d1 = struct.unpack('<II', f.read(8))
        off = struct.unpack('<Q', f.read(8))[0]
        sz = struct.unpack('<Q', f.read(8))[0]
        if i < 3 or i == 399:
            print(f"[{i}] {name}: off={off} sz={sz}")
    data_start = f.tell()
    print(f"Weight data starts at file offset: {data_start}")
    print(f"First tensor off=0 -> absolute file offset = {data_start}")
    
    # Read scales at data_start
    f.seek(data_start)
    raw = f.read(16)
    print(f"First 16 bytes at data_start: {' '.join(f'{b:02x}' for b in raw)}")
    
    # Read packed data
    f.seek(data_start + 1024)
    packed = f.read(16)
    print(f"Packed data at +1024: {' '.join(f'{b:02x}' for b in packed)}")
    
    # The C++ loader computed abs_offset = p - map_ + off
    # where p is the position AFTER reading this tensor's index entry
    # and off = 0 for the first tensor
    # So abs_offset = file_pos_after_reading_index_entry + 0
    # Let's verify this matches data_start
    
    # Re-read the first index to get p position
    f.seek(256)
    nl = struct.unpack('<I', f.read(4))[0]
    name = f.read(nl).decode()
    null = f.read(1)
    ndim = struct.unpack('<I', f.read(4))[0]
    d0, d1 = struct.unpack('<II', f.read(8))
    off = struct.unpack('<Q', f.read(8))[0]
    pos_after = f.tell()
    off = struct.unpack('<Q', f.read(8))[0]
    pos_after = f.tell()
    abs_pos = pos_after + 0  # off = 0 for first tensor
    print(f"After reading first index: {pos_after}")
    print(f"C++ abs_offset = {pos_after} + {off} = {pos_after + off}")
    print(f"Actual data start: {data_start}")
    print(f"Match: {pos_after + off == data_start}")
