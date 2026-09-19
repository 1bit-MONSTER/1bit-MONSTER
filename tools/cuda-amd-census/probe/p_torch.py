# p_torch.py -- the framework-level surface. This is *the* row that matters for
# most real workloads: a CUDA-shaped tensor API actually working on AMD.
#
# On Linux the honest mapping is not "CUDA -> ROCm via a shim" but
# "torch shipped as a ROCm/HIP build" (torch.version.hip set, device 'cuda'
# aliased onto the HIP backend). A CPU-only torch install reports cuda
# unavailable and must NOT be recorded as a GPU capability, no matter how
# tempting the 'cuda' device alias is.
import sys


def emit(surface, result, detail):
    print("PROBE|%s|%s|%s" % (surface, result, detail))
    sys.stdout.flush()


def main():
    surface = "torch"
    try:
        import torch
    except Exception as e:
        emit(surface, "UNSUPPORTED", "import torch failed: %s: %s" % (type(e).__name__, e))
        return 3

    ver = torch.__version__
    hip = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)

    try:
        avail = bool(torch.cuda.is_available())
    except Exception as e:
        emit(surface, "ERROR", "torch.cuda.is_available() raised: %s" % e)
        return 5

    # A "+cpu" wheel is not a GPU build at all: UNSUPPORTED, not PASS.
    if not avail:
        emit(surface, "UNSUPPORTED",
             "torch %s is not a GPU build (hip=%s cuda=%s is_available=False)"
             % (ver, hip, cuda_ver))
        return 3

    try:
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", None) or getattr(props, "name", "?")

        # independent CPU reference matmul, fp32
        torch.manual_seed(1234)
        n = 512
        a = torch.randn(n, n, dtype=torch.float32)
        b = torch.randn(n, n, dtype=torch.float32)
        ref = a @ b
        got = ((a.to(dev)) @ (b.to(dev))).to("cpu")
        max_abs = float((got - ref).abs().max())

        # GPU-execution witness: the host never writes this tensor's values,
        # so matching it means device-side code produced it.
        want = [float(i) * 3.0 for i in range(8)]
        g = (torch.arange(0, 8, dtype=torch.float32, device=dev) * 3.0).to("cpu").tolist()

        if max_abs > 1e-2:
            emit(surface, "INCORRECT",
                 "matmul 512^2 max_abs=%.3e on %s (torch %s)" % (max_abs, name, ver))
            return 4
        if g != want:
            emit(surface, "INCORRECT",
                 "device arange witness wrong: %r expected %r" % (g, want))
            return 4

        emit(surface, "PASS",
             "torch %s hip=%s cuda=%s dev=\"%s\" arch=%s matmul512 max_abs=%.2e witness_ok"
             % (ver, hip, cuda_ver, name, arch, max_abs))
        return 0
    except Exception as e:
        emit(surface, "ERROR", "torch GPU path raised: %s: %s" % (type(e).__name__, e))
        return 5


if __name__ == "__main__":
    sys.exit(main())
