# FlashOCC XPU Installation Guide

Complete installation guide for FlashOCC on Intel XPU (DG2/Arc GPUs).

## Prerequisites

### Hardware
- Intel Arc GPU (DG2) - A380, A750, A770, etc.
- At least 16GB system RAM
- 8GB+ GPU memory recommended

### Software
- Ubuntu 20.04/22.04 or similar Linux distribution
- Intel oneAPI Base Toolkit 2025.2.0+
- Python 3.8-3.10
- Conda or Miniconda

## Installation Steps

### Step 1: Install Intel GPU Drivers and oneAPI

```bash
# Install Intel GPU drivers
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
    https://repositories.intel.com/graphics/ubuntu jammy arc' | \
    sudo tee /etc/apt/sources.list.d/intel-graphics.list
sudo apt update
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero

# Install Intel oneAPI Base Toolkit
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
    https://apt.repos.intel.com/oneapi all main" | \
    sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit-2025.2.0
```

### Step 2: Create Python Environment

```bash
# Create conda environment
conda create -n flashocc-xpu python=3.10
conda activate flashocc-xpu

# Install PyTorch XPU (must be installed first!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Verify PyTorch XPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}')"
```

### Step 3: Install Dependencies

```bash
# Install compatible numpy (for numba)
pip install numpy==1.21.6

# Install mmdetection dependencies (without mmcv-full)
pip install addict
pip install yapf==0.40.1
pip install opencv-python
pip install Pillow
pip install terminaltables

# Install other dependencies
pip install lyft_dataset_sdk
pip install networkx==2.2
pip install numba==0.56.4 #xpu, 0.55.0
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39

pip install pyyaml
pip install PyYAML

# Load oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Set compiler environment variables
export CXX=$(which icpx)
export CC=$(which icx)


git clone https://github.com/xzhan34/mmcv.git -b xpu

export MMCV_WITH_OPS=1 
export FORCE_XPU=1
export TORCH_CUDA_ARCH_LIST="XPU"
export CXX=$(which icpx)
export CC=$(which icx)
export PYTORCH_BUILD_VERSION_GE_1_13=1
export TORCH_EXTENSIONS_IGNORE_ABI_COMPATIBILITY=1
CC=icx CXX=icpx python -m pip install -e . --no-build-isolation --no-deps
export PYTHONPATH=`pwd`:$PYTHONPATH

python -c "import mmcv; import mmcv._ext; print('mmcv._ext loaded')"

# Disable isolation and point CC/CXX at the oneAPI toolchain so the SYCL ops compile against torch XPU.
# The setup hook copies the compiled `_ext` library back into mmcv/ for development use.


**Important**: DO NOT install `mmcv-full`, `mmdet`, or `mmsegmentation` yet. These require CUDA and won't work with XPU.

### Step 4: Clone and Setup FlashOCC

```bash
# Clone FlashOCC
cd ~/work
git clone https://github.com/Yzichen/FlashOCC.git
cd FlashOCC

# Clone mmdetection3d (compatible version)
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
pip install -e . --no-build-isolation --no-deps
cd ..
```

### Step 5: Build XPU Operators

```bash
# Load oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Set compiler environment variables
export CXX=$(which icpx)
export CC=$(which icx)

# Build main operators (bev_pool, nearest_assign)
cd projects
pip install -e . --no-build-isolation --no-deps
#python setup_xpu.py build_ext --inplace

# Build DVR operator
cd ../lib/dvr
pip install -e . --no-build-isolation --no-deps
#python setup_xpu.py build_ext --inplace
python -m pip install -e .--no-deps --no-build-isolation


bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth 1 --eval map

# Return to root
cd ../..
```

### Step 6: Verify Installation

```bash
# Test XPU availability
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}'); print(f'Device count: {torch.xpu.device_count()}')"

# Test operators loading (simple test without mmdet3d imports)
cd projects
python -c "
import sys
import torch
# Test individual operator imports
from mmdet3d_plugin.ops.bev_pool_v2.bev_pool_ext import bev_pool_v2_forward
from mmdet3d_plugin.ops.bev_pool.bev_pool_ext import bev_max_pool_forward
from mmdet3d_plugin.ops.nearest_assign.nearest_assign_ext import nearest_assign_forward
print('✅ All XPU operators loaded successfully!')
"
```

### Step 7: Prepare nuScenes Dataset

Follow the original instructions:

```bash
# Download nuScenes dataset to data/nuscenes/
# Structure:
# FlashOCC/
#   data/
#     nuscenes/
#       v1.0-trainval/
#       sweeps/
#       samples/

# Create dataset info
python tools/create_data_bevdet.py

# Download occupancy GT from CVPR2023-3D-Occupancy-Prediction
# Place in data/nuscenes/gts/
```

## Alternative: Minimal Installation (Operators Only)

If you only need to build and test the XPU operators without the full training pipeline:

```bash
conda create -n flashocc-xpu-minimal python=3.10
conda activate flashocc-xpu-minimal

# Install PyTorch XPU
pip install torch --index-url https://download.pytorch.org/whl/xpu

# Load oneAPI and build
source /opt/intel/oneapi/setvars.sh
export CXX=$(which icpx)
export CC=$(which icx)

cd FlashOCC/projects
python setup_xpu.py build_ext --inplace
pip install -v -e . --no-build-isolation --no-deps

cd ../lib/dvr
python setup_xpu.py build_ext --inplace
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'mmcv._ext'

**Cause**: Pre-built `mmcv-full` wheel is built for CUDA, not XPU.

**Solution**:
- DO NOT install `mmcv-full` for XPU development
- For inference only, the XPU operators are sufficient
- For full training, mmdet3d needs to be adapted to work without mmcv CUDA ops

### Issue: mmdet/mmdet3d import errors

**Cause**: mmdetection and mmdetection3d expect CUDA environment.

**Solution**:
- For operator testing: Import operators directly without going through mmdet3d
- For inference: You may need to stub out or adapt mmdet3d CUDA dependencies
- For training: Full port of mmdet3d to XPU is required (beyond operator migration)

### Issue: CUDA_HOME not set

**Cause**: Using `setup.py` instead of `setup_xpu.py`.

**Solution**: Always use `setup_xpu.py` for XPU builds.

### Issue: Compilation errors

See [COMPILATION_FIXES.md](../COMPILATION_FIXES.md) and [BUILD_XPU_GUIDE.md](../BUILD_XPU_GUIDE.md) for detailed troubleshooting.

## Quick Test Script

Create a test file `test_xpu_ops.py`:

```python
#!/usr/bin/env python
import torch
import sys

# Add projects to path
sys.path.insert(0, 'projects')

print("Testing XPU availability...")
if not torch.xpu.is_available():
    print("❌ XPU not available!")
    sys.exit(1)
print(f"✅ XPU available: {torch.xpu.device_count()} device(s)")

print("\nTesting operator imports...")
try:
    from mmdet3d_plugin.ops.bev_pool_v2.bev_pool_ext import bev_pool_v2_forward
    print("✅ bev_pool_v2 loaded")
except Exception as e:
    print(f"❌ bev_pool_v2 failed: {e}")

try:
    from mmdet3d_plugin.ops.bev_pool.bev_pool_ext import bev_max_pool_forward
    print("✅ bev_max_pool loaded")
except Exception as e:
    print(f"❌ bev_max_pool failed: {e}")

try:
    from mmdet3d_plugin.ops.nearest_assign.nearest_assign_ext import nearest_assign_forward
    print("✅ nearest_assign loaded")
except Exception as e:
    print(f"❌ nearest_assign failed: {e}")

print("\n✅ All operators loaded successfully!")
```

Run with:
```bash
python test_xpu_ops.py
```

## Next Steps

1. **For operator development**: You're ready! Test operators with synthetic data.

2. **For inference**: Adapt the inference pipeline to work without mmcv CUDA ops.

3. **For training**: Additional work needed to port mmdet3d training pipeline to XPU.

## Reference Documents

- [README_XPU.md](../README_XPU.md) - Complete XPU migration overview
- [BUILD_XPU_GUIDE.md](../BUILD_XPU_GUIDE.md) - Quick build guide
- [COMPILATION_FIXES.md](../COMPILATION_FIXES.md) - Compilation issue fixes
- [CUDA_TO_SYCL_MIGRATION.md](../CUDA_TO_SYCL_MIGRATION.md) - Technical migration details
