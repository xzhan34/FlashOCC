"""
DVR (Depth Volume Rendering) XPU Setup Script

SYCL implementation for Intel XPU
Based on Intel oneAPI Base Toolkit 2025.2.0
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

def make_dvr_xpu_extension():
    """
    Create DVR XPU extension with SYCL compiler
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
        '-DMAX_D=1446',
        '-DMAX_STEP=1000',
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
        name='dvr',
        sources=[
            'dvr.cpp',
            'dvr_sycl.cpp',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=['-fsycl'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    )

if __name__ == '__main__':
    setup(
        name='dvr',
        version='1.0.0',
        description='Depth Volume Rendering with Intel XPU Support (SYCL)',
        author='FlashOCC Team',
        ext_modules=[make_dvr_xpu_extension()],
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=False)
        },
        zip_safe=False,
    )
    
    print("\n" + "="*60)
    print("DVR XPU Extension Build Completed!")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Ray marching with voxel traversal")
    print("  ✓ Forward rendering (inference)")
    print("  ✓ Backward rendering (training with gradients)")
    print("  ✓ Occupancy grid initialization")
    print("\nSYCL Implementation Details:")
    print("  - Uses atomic operations for gradient accumulation")
    print("  - Supports L1, L2, and AbsRel loss types")
    print("  - Compatible with Intel DG2 GPU")
    print("="*60)
