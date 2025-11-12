"""
FlashOCC XPU Setup Script

Intel DG2 platform support with SYCL implementation
Uses PyTorch native XPU support (NO IPEX dependency required)
Based on Intel oneAPI Base Toolkit 2025.2.0
"""

from setuptools import find_packages, setup

import os
import shutil
import sys
import torch
import warnings
from os import path as osp
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

def make_xpu_extension(name, module, sources):
    """
    Create XPU extension with SYCL compiler
    Uses Intel DPC++/C++ Compiler 2025.2.0
    """

    # Validate PyTorch XPU support
    try:
        import torch
        if not hasattr(torch, 'xpu'):
            raise RuntimeError(
                "PyTorch XPU support not found!\n"
                "Install with: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu\n"
                "Requires: Intel oneAPI Base Toolkit 2025.2.0"
            )
    except ImportError:
        raise RuntimeError(
            "PyTorch not installed!\n"
            "Install with: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu"
        )

    # Set SYCL compiler
    oneapi_root = os.environ.get('ONEAPI_ROOT', '/opt/intel/oneapi')
    sycl_compiler = os.path.join(oneapi_root, 'compiler/latest/bin/icpx')

    # Override CXX compiler to use SYCL
    os.environ['CXX'] = sycl_compiler

    # SYCL compilation flags for Intel DG2
    extra_compile_args = [
        '-std=c++17',
        '-O3',
        '-fPIC',
        '-Wno-deprecated-declarations',
        '-fsycl',
        '-fsycl-targets=spir64',
    ]

    # Include paths for oneAPI 2025.2.0
    include_dirs = [
        os.path.join(oneapi_root, 'compiler/latest/include'),
        os.path.join(oneapi_root, 'compiler/latest/include/sycl'),
    ]

    # Library paths
    library_dirs = [
        os.path.join(oneapi_root, 'compiler/latest/lib'),
    ]

    # Add SYCL libraries
    libraries = ['sycl']

    return CppExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args=extra_compile_args,
        extra_link_args=['-fsycl'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    )

def check_environment():
    """Verify oneAPI environment is configured"""

    oneapi_root = os.environ.get('ONEAPI_ROOT', '/opt/intel/oneapi')

    if not os.path.exists(oneapi_root):
        print(f"WARNING: oneAPI not found at {oneapi_root}")
        print("Install Intel oneAPI Base Toolkit 2025.2.0:")
        print("  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html")
        print("\nOn Linux, run:")
        print("  source /opt/intel/oneapi/setvars.sh")
        print("\nOn Windows, run:")
        print("  C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat")
        return False

    # Check for SYCL compiler
    dpcpp_path = os.path.join(oneapi_root, 'compiler/latest/bin/icpx')
    if os.name == 'nt':
        dpcpp_path = os.path.join(oneapi_root, 'compiler\\latest\\bin\\icx.exe')

    if not os.path.exists(dpcpp_path):
        print(f"WARNING: SYCL compiler not found at {dpcpp_path}")
        return False

    print(f" Found oneAPI at: {oneapi_root}")
    print(f" Found SYCL compiler: {dpcpp_path}")
    return True

if __name__ == '__main__':
    # Verify environment
    env_ok = check_environment()
    if not env_ok:
        print("\nEnvironment check failed. Build may not succeed.")
        print("Continue anyway? (Ctrl+C to cancel)")
        input()

    # Build XPU extensions
    setup(
        name='FlashOCC-XPU',
        version='2.0.0',
        description='FlashOCC with Intel XPU Support (Native PyTorch + SYCL)',
        author='FlashOCC Team',
        packages=find_packages(),
        ext_modules=[
            # BEV pooling extension (sum + max, SYCL)
            make_xpu_extension(
                name='bev_pool_ext',
                module='mmdet3d_plugin.ops.bev_pool',
                sources=[
                    'src/bev_pooling.cpp',
                    'src/bev_sum_pool.cpp',
                    'src/bev_sum_pool_sycl.cpp',
                    'src/bev_max_pool.cpp',
                    'src/bev_max_pool_sycl.cpp',
                ],
            ),
            # BEV pooling v2 extension (SYCL)
            make_xpu_extension(
                name='bev_pool_v2_ext',
                module='mmdet3d_plugin.ops.bev_pool_v2',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_sycl.cpp',
                ],
            ),
            # Nearest assignment extension (SYCL)
            make_xpu_extension(
                name='nearest_assign_ext',
                module='mmdet3d_plugin.ops.nearest_assign',
                sources=[
                    'src/nearest_assign.cpp',
                    'src/nearest_assign_sycl.cpp',
                ],
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=False)
        },
        zip_safe=False,
    )

    print("\n" + "="*60)
    print("Build completed successfully!")
    print("="*60)
    print("\nAll CUDA operators have been migrated to SYCL:")
    print("  ✓ bev_pool_v2    - Pillar pooling with depth weighting")
    print("  ✓ bev_max_pool   - Max pooling for BEV features")
    print("  ✓ bev_sum_pool   - Sum pooling for BEV features")
    print("  ✓ nearest_assign - Nearest neighbor assignment")
    print("  ✓ dvr (lib/dvr)  - Depth Volume Rendering")
    print("\nVerify installation:")
    print("  python tools/check_xpu_installation.py")
    print("\nRun training:")
    print("  python tools/train_xpu.py configs/flashocc/flashocc-r50.py")
    print("="*60)
