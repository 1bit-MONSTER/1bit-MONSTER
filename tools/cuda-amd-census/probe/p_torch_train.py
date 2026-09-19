#!/usr/bin/env python3
"""p_torch_train.py — Step 5 of goal mu7vzirt-7k1q97: the integration-validated rung.

The census validated ONE operation per library (a single SGEMM, one FFT, one
conv). That is evidence the libraries answer correctly; it is NOT evidence that
the machine can run a workload. This moves one rung up the census's own evidence
ladder: a real application completing end-to-end.

It trains a small MLP on GPU and reports four independent things:

  1. device      — torch sees the AMD GPU, with its arch named
  2. execution   — a real forward + backward + optimizer step ran, and a
                   device-side witness (a value the host cannot synthesise)
                   proves it was not a CPU fallback
  3. gradients   — the GPU's gradients match a CPU reference for the SAME
                   initial parameters and the SAME batch, to fp32 tolerance.
                   The CPU and CUDA/HIP paths are different kernels, so
                   agreement is evidence rather than a tautology.
  4. convergence — loss actually falls over a short training run, which no
                   single-op probe can show.

Fail closed: a CPU fallback, a missing device witness, mismatched gradients or a
flat loss all produce a non-PASS verdict rather than a hopeful one.
"""
import sys


def emit(surface, result, detail):
    print("PROBE|%s|%s|%s" % (surface, result, detail))
    sys.stdout.flush()


def main():
    surface = "torch_train"
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        emit(surface, "UNSUPPORTED", "import torch failed: %s: %s" % (type(e).__name__, e))
        return 3

    if not torch.cuda.is_available():
        emit(surface, "UNSUPPORTED",
             "torch %s is not a GPU build (hip=%s, is_available=False)"
             % (torch.__version__, getattr(torch.version, "hip", None)))
        return 3

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "?")

    # ---- 2. device-execution witness -----------------------------------------
    # Host cannot produce these: they are computed by device code.
    w = torch.arange(0, 8, dtype=torch.float32, device=dev) * 3.0 + 1.0
    want = [float(i * 3 + 1) for i in range(8)]
    if w.to("cpu").tolist() != want:
        emit(surface, "INCORRECT", "device witness wrong: %r" % w.to("cpu").tolist())
        return 4

    torch.manual_seed(1234)

    # ---- a small MLP and a synthetic regression task -------------------------
    D_IN, D_H, N = 8, 32, 256
    X = torch.randn(N, D_IN)
    Y = (torch.sin(X[:, 0:1]) + X[:, 1:2] ** 2 - 0.5 * X[:, 2:3]).float()

    def make_model():
        torch.manual_seed(7)          # identical init on both sides
        return nn.Sequential(nn.Linear(D_IN, D_H), nn.Tanh(), nn.Linear(D_H, 1))

    def loss_of(m, x, y):
        return nn.functional.mse_loss(m(x), y)

    # ---- 3. gradient cross-check: same params, same batch, CPU vs GPU ---------
    cpu_model = make_model().double()
    gpu_model = make_model().double().to(dev)
    gpu_model.load_state_dict(cpu_model.state_dict())

    lc = loss_of(cpu_model, X.double(), Y.double())
    lc.backward()
    cpu_grads = [p.grad.clone() for p in cpu_model.parameters()]

    lg = loss_of(gpu_model, X.double().to(dev), Y.double().to(dev))
    lg.backward()
    gpu_grads = [p.grad.to("cpu").clone() for p in gpu_model.parameters()]

    loss_diff = abs(float(lc) - float(lg))
    gmax = 0.0
    for a, b in zip(cpu_grads, gpu_grads):
        gmax = max(gmax, float((a - b).abs().max()))
    GTOL = 1e-6
    if loss_diff > GTOL or gmax > GTOL:
        emit(surface, "INCORRECT",
             "CPU/GPU disagree: loss_diff=%.3e grad_max_abs=%.3e on %s" % (loss_diff, gmax, name))
        return 4

    # ---- 4. actually train on the GPU ---------------------------------------
    torch.manual_seed(11)
    model = make_model().float().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    xd, yd = X.to(dev), Y.to(dev)

    losses = []
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xd), yd)
        loss.backward()
        opt.step()
        losses.append(float(loss))

    first = sum(losses[:10]) / 10.0
    last = sum(losses[-10:]) / 10.0

    # params must be on the GPU and must have moved
    p0 = next(model.parameters())
    if not p0.is_cuda:
        emit(surface, "INCORRECT", "trained parameters are not on the GPU")
        return 4
    if not (last < first * 0.5):
        emit(surface, "INCORRECT",
             "loss did not fall: first10=%.6f last10=%.6f on %s" % (first, last, name))
        return 4

    emit(surface, "PASS",
         "torch %s hip=%s | dev=\"%s\" arch=%s | witness_ok | "
         "cpu_vs_gpu loss_diff=%.2e grad_max_abs=%.2e (200 Adam steps, 2-layer MLP %d->%d->1) | "
         "loss %.6f -> %.6f"
         % (torch.__version__, getattr(torch.version, "hip", None), name, arch,
            loss_diff, gmax, D_IN, D_H, first, last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
