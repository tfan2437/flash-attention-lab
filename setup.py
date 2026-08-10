"""Builds flash_lab._C when a CUDA toolkit is available; otherwise installs the Python package only.

Set FLASH_LAB_BUILD_CUDA=0/1 to force the choice. FLASH_LAB_PTXAS_VERBOSE=1 prints register and
shared-memory usage per kernel, FLASH_LAB_DEBUG=1 synchronizes after every launch.
"""

import glob
import os

from setuptools import setup

ROOT = os.path.dirname(os.path.abspath(__file__))


def want_cuda() -> bool:
    flag = os.environ.get("FLASH_LAB_BUILD_CUDA")
    if flag is not None:
        return flag == "1"
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return False
    return CUDA_HOME is not None


ext_modules = []
cmdclass = {}

if want_cuda():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0")

    sources = sorted(glob.glob("csrc/**/*.cpp", recursive=True))
    sources += sorted(glob.glob("csrc/**/*.cu", recursive=True))

    cxx_flags = ["-O3", "-std=c++17"]
    nvcc_flags = ["-O3", "-std=c++17", "-lineinfo"]
    if os.environ.get("FLASH_LAB_PTXAS_VERBOSE") == "1":
        nvcc_flags.append("-Xptxas=-v")
    if os.environ.get("FLASH_LAB_DEBUG") == "1":
        cxx_flags.append("-DFLASH_LAB_DEBUG")
        nvcc_flags.append("-DFLASH_LAB_DEBUG")

    ext_modules.append(
        CUDAExtension(
            name="flash_lab._C",
            sources=sources,
            include_dirs=[os.path.join(ROOT, "csrc")],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(ext_modules=ext_modules, cmdclass=cmdclass)
