"""Build and runtime helpers for the strict Qwen3 decode-layer cases."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

EXPERIMENT_DIR = Path(__file__).parent.resolve()
TORCH2AIE_ROOT = EXPERIMENT_DIR.parents[1]
TOOLCHAIN_DIR = Path(os.environ.get("TORCH2AIE_TOOLCHAIN", TORCH2AIE_ROOT / "toolchain")).resolve()
AIETOOLS_DIR = Path(os.environ.get("AIETOOLS_DIR", TOOLCHAIN_DIR / "aietools")).resolve()
MLIR_AIE_DIR = Path(os.environ.get("MLIR_AIE_DIR", TOOLCHAIN_DIR / "mlir_aie")).resolve()
XRT_DIR = Path(os.environ.get("XRT_DIR", TOOLCHAIN_DIR / "xrt")).resolve()
SYSROOT_DIR = Path(os.environ.get("TORCH2AIE_SYSROOT", TOOLCHAIN_DIR / "sysroot")).resolve()
ROLE_OBJECT_DIR = TORCH2AIE_ROOT / "build" / "qwen3_decode_layer_objects"
COMPILE_DIR = TORCH2AIE_ROOT / "build" / "qwen3_decode_layer_compile"
AIECC_JOBS = os.environ.get("AIECC_JOBS", "4")
MAIN16_KERNEL_SOURCE = os.environ.get("QWEN3_MAIN16_KERNEL_SOURCE", "qwen3_decode_kernels.cc")
EDGE_ATTENTION_SOURCE = os.environ.get("QWEN3_EDGE_ATTENTION_SOURCE", "edge_attention.cc")
FULL_VECTOR_STATION_SOURCE = os.environ.get("QWEN3_FULL_VECTOR_STATION_SOURCE", "full_vector_station.cc")
POSTPROCESS_QKV_SOURCE = os.environ.get("QWEN3_POSTPROCESS_QKV_SOURCE", "postprocess_qkv.cc")
SWIGLU_SOURCE = os.environ.get("QWEN3_SWIGLU_SOURCE", "swiglu.cc")

ROLE_KERNEL_SOURCES = {
    "edge_attention.o": (EDGE_ATTENTION_SOURCE,),
    "edge_attention_profile.o": ("edge_attention_profile.cc", "edge_attention.cc"),
    "full_vector_station.o": (FULL_VECTOR_STATION_SOURCE,),
    "full_vector_station_profile.o": ("full_vector_station_profile.cc", "full_vector_station.cc"),
    "main16_q4nx_section_cell_probe.o": (
        "probes/main16_q4nx_section_cell_probe.cc",
    ),
    "main16_q4nx_chunk_slot_probe.o": (
        "probes/main16_q4nx_chunk_slot_probe.cc",
    ),
    "main16_q4nx_slot_window_probe.o": (
        "probes/main16_q4nx_slot_window_probe.cc",
    ),
    "main16_q4nx_steady_cell_probe.o": (
        "probes/main16_q4nx_steady_cell_probe.cc",
    ),
    "main_projection_q4nx_fast.o": (MAIN16_KERNEL_SOURCE,),
    "main_projection_q4nx_profile.o": ("qwen3_decode_kernels_profile.cc", "qwen3_decode_kernels.cc"),
    "postprocess_qkv.o": (POSTPROCESS_QKV_SOURCE,),
    "swiglu.o": (SWIGLU_SOURCE,),
}
PRODUCTION_ROLE_OBJECTS = (
    "edge_attention.o",
    "full_vector_station.o",
    "main_projection_q4nx_fast.o",
    "postprocess_qkv.o",
    "swiglu.o",
)
ROLE_KERNEL_HEADERS = ("qwen3_constants.h", "record_format.h")
LINK_WITH_RE = re.compile(r'link_with = "[^"]*/([^/"]+\.o)"')


class BuildHasher(Protocol):
    def update(self, data: bytes, /) -> None:
        ...


def _prepend_env(name: str, paths: list[Path | str]) -> None:
    existing = os.environ.get(name, "")
    values = [str(path) for path in paths if str(path)]
    os.environ[name] = ":".join(values + ([existing] if existing else []))


def configure_environment() -> None:
    _prepend_env(
        "PATH",
        [
            TOOLCHAIN_DIR / "bin",
            MLIR_AIE_DIR / "bin",
            XRT_DIR / "bin",
            "/usr/local/bin",
            "/usr/bin",
            AIETOOLS_DIR / "bin",
        ],
    )
    _prepend_env(
        "LD_LIBRARY_PATH",
        [
            SYSROOT_DIR / "usr/lib64",
            TOOLCHAIN_DIR / "mlir_aie.libs",
            MLIR_AIE_DIR / "lib",
            AIETOOLS_DIR / "lib",
            AIETOOLS_DIR / "lib/lnx64.o",
            XRT_DIR / "lib64",
        ],
    )


configure_environment()


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def _role_object_path(object_name: str) -> Path:
    return ROLE_OBJECT_DIR / object_name


def _compile_aie_object(source_names: tuple[str, ...], object_name: str) -> None:
    if not source_names:
        raise ValueError(f"missing source for role object: {object_name}")
    src = EXPERIMENT_DIR / source_names[0]
    obj = _role_object_path(object_name)
    obj.parent.mkdir(parents=True, exist_ok=True)
    compiler = TOOLCHAIN_DIR / "bin" / "xchesscc_wrapper"
    cmd = [
        str(compiler),
        "aie2p",
        f"-I{EXPERIMENT_DIR}",
        f"-I{AIETOOLS_DIR / 'include'}",
        f"-I{MLIR_AIE_DIR / 'include'}",
        f"-I{MLIR_AIE_DIR / 'include/aie_kernels'}",
        f"-I{MLIR_AIE_DIR / 'include/aie_kernels/aie2p'}",
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    print(f"  Compiling {object_name} with Chess from {src.name}...")
    run_command(cmd)


def _linked_role_objects(mlir_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    objects: list[str] = []
    for object_name in LINK_WITH_RE.findall(mlir_text):
        if object_name not in ROLE_KERNEL_SOURCES:
            raise ValueError(f"unknown AIE role object in link_with: {object_name}")
        if object_name not in seen:
            seen.add(object_name)
            objects.append(object_name)
    return tuple(objects)


def _compile_linked_role_objects(object_names: tuple[str, ...]) -> None:
    for object_name in object_names:
        _compile_aie_object(ROLE_KERNEL_SOURCES[object_name], object_name)


def _rewrite_link_with_paths(mlir_text: str, object_names: tuple[str, ...]) -> str:
    rewritten = mlir_text
    for object_name in object_names:
        rewritten = re.sub(
            rf'link_with = "[^"]*/{re.escape(object_name)}"',
            f'link_with = "{object_name}"',
            rewritten,
        )
    return rewritten


def _stage_role_objects(object_names: tuple[str, ...]) -> None:
    COMPILE_DIR.mkdir(parents=True, exist_ok=True)
    for object_name in object_names:
        shutil.copy2(_role_object_path(object_name), COMPILE_DIR / object_name)


def _update_build_key_file(hasher: BuildHasher, path: Path) -> None:
    hasher.update(path.name.encode())
    hasher.update(b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")


def _build_key(mlir_text: str, object_names: tuple[str, ...], command: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"torch2aie-qwen3-decode-layer-chess-v2\0")
    hasher.update("\n".join(command).encode())
    hasher.update(b"\0")
    hasher.update(mlir_text.encode())
    hasher.update(b"\0")
    for header_name in ROLE_KERNEL_HEADERS:
        _update_build_key_file(hasher, EXPERIMENT_DIR / header_name)
    for object_name in object_names:
        for source_name in ROLE_KERNEL_SOURCES[object_name]:
            _update_build_key_file(hasher, EXPERIMENT_DIR / source_name)
    return hasher.hexdigest()


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> None:
    mlir_text = mlir_path.read_text()
    object_names = _linked_role_objects(mlir_text)
    chess_mlir_text = _rewrite_link_with_paths(mlir_text, object_names)
    COMPILE_DIR.mkdir(parents=True, exist_ok=True)
    chess_mlir_path = COMPILE_DIR / "design.mlir"
    chess_mlir_path.write_text(chess_mlir_text)
    aiecc = TOOLCHAIN_DIR / "bin" / "aiecc"
    cmd = [
        str(aiecc),
        "-v",
        f"-j{AIECC_JOBS}",
        f"--aietools={AIETOOLS_DIR}",
        "--no-compile-host",
        "--alloc-scheme=basic-sequential",
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(chess_mlir_path),
    ]
    key_path = xclbin_path.with_suffix(".buildkey")
    build_key = _build_key(chess_mlir_text, object_names, cmd)
    if xclbin_path.exists() and insts_path.exists() and key_path.exists() and key_path.read_text() == build_key:
        print(f"  Reusing cached NPU build: {xclbin_path}")
        return
    _compile_linked_role_objects(object_names)
    _stage_role_objects(object_names)
    print("  Compiling MLIR-AIE with Chess-enabled aiecc...")
    xclbin_path.parent.mkdir(parents=True, exist_ok=True)
    insts_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(COMPILE_DIR / "design.mlir.prj", ignore_errors=True)
    run_command(cmd, cwd=COMPILE_DIR)
    key_path.write_text(build_key)


def load_kernel(xclbin_path: Path, insts_path: Path):
    import aie.utils as aie_utils
    from aie.utils.npukernel import NPUKernel

    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    return aie_utils.DefaultNPURuntime.load(kernel)


def device() -> object:
    import aie.utils as aie_utils

    return aie_utils.DefaultNPURuntime.device()


def run(handle: object, buffers: list[object]) -> object:
    import aie.utils as aie_utils

    return aie_utils.DefaultNPURuntime.run(handle, buffers)


def cleanup() -> None:
    import aie.utils as aie_utils

    aie_utils.DefaultNPURuntime.cleanup()
