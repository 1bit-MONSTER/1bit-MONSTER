# onebit_smoke.mojo — Mojo 1.0 drives the 1bit engine through the C ABI seam.
# Build:  mojo build tests/mojo/onebit_smoke.mojo
# Run:    ./onebit_smoke <weights_dir> [model_name]
# Proves the fold: Mojo dlopens libonebit.so (OwnedDLHandle), calls
# extern "C" functions, and runs a real decode loop through the engine.

from std.ffi import OwnedDLHandle, c_int, c_char
from std.memory import Pointer
from std.sys import argv


def read_cstr(p: Pointer[UInt8, MutUntrackedOrigin]) -> String:
    """Read a NUL-terminated C string into a Mojo String."""
    var len = 0
    while p.unsafe_load(len) != 0:
        len += 1
    return String(
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=p, length=len
        )
    )

def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: onebit_smoke <weights_dir> [model_name]")
        return
    var weights_dir = String(args[1])
    var model_name = String(args[2]) if len(args) > 2 else String()

    var lib = OwnedDLHandle("build/libonebit.so")

    # ── version ──
    var version = lib.get_function[Pointer[UInt8, MutUntrackedOrigin]](
        "onebit_version"
    )
    print("1bit version:", read_cstr(version()))

    # ── lifecycle ──
    var create = lib.get_function[Pointer[NoneType, MutUntrackedOrigin]](
        "onebit_create"
    )
    var h = create()

    var last_error = lib.get_function[Pointer[UInt8, MutUntrackedOrigin]](
        "onebit_last_error"
    )

    # ── init ──
    var init = lib.get_function[c_int]("onebit_init")
    var wd = weights_dir
    var mn = model_name
    var rc = init(
        h,
        wd.as_c_string_slice().unsafe_ptr(),
        mn.as_c_string_slice().unsafe_ptr(),
    )
    if rc != 0:
        print("init failed rc=", rc, "err:", read_cstr(last_error(h)))
        lib.get_function[NoneType]("onebit_destroy")(h)
        return
    print("init ok")

    # ── backends ──
    var count = Int(lib.get_function[c_int]("onebit_backend_count")(h))
    print("backend_count:", count)
    var get_id = lib.get_function[Pointer[UInt8, MutUntrackedOrigin]](
        "onebit_backend_id"
    )
    for i in range(count):
        print("  backend[", i, "]:", read_cstr(get_id(h, c_int(i))))

    # ── decode loop: BOS → 8 tokens ──
    var generate = lib.get_function[c_int]("onebit_generate")
    var token = c_int(1)  # BOS
    for _ in range(8):
        var next = generate(h, token)
        if next < 0:
            print(
                "generate error:",
                read_cstr(last_error(h)),
            )
            break
        print("token:", next)
        token = next

    # ── reset + health ──
    var reset = lib.get_function[c_int]("onebit_reset")
    print("reset:", reset(h))
    var health = lib.get_function[c_int]("onebit_health_check")
    print("health:", health(h))

    # ── destroy ──
    lib.get_function[NoneType]("onebit_destroy")(h)
    print("done")
