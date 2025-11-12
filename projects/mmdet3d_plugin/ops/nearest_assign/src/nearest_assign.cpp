// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
// Modified for Intel XPU with SYCL
#include <torch/torch.h>
#include <sycl/sycl.hpp>

using namespace sycl;

// SYCL function declarations
void nearest_assign(const int* l2s_key,
                    int l2s_size,
                    const int *occind2detind,
                    int inst_size,
                    const int *occ_pred,
                    const int *inst_xyz,
                    const int *inst_cls,
                    const int *inst_id_list,
                    int* inst_pred,
                    sycl::queue& q);

void nearest_assign_forward(
  const at::Tensor _occ_pred,    // (200, 200, 16)
  const at::Tensor _l2s_key,     // (l2s_size, 1)
  const at::Tensor _occind2detind, // (10, 1)
  const at::Tensor _inst_cls,     // (inst_size, 1)
  const at::Tensor _inst_xyz,     // (inst_size, 3)
  const at::Tensor _inst_id_list, // (inst_size, 1)
  at::Tensor _inst_pred           // (200, 200, 16)
) {
  int l2s_size = _l2s_key.size(0);
  int inst_size = _inst_xyz.size(0);

  // Get XPU device and create SYCL queue
  auto device = _occ_pred.device();
  TORCH_CHECK(device.is_xpu(), "Tensors must be on XPU device");
  sycl::queue q = sycl::queue(sycl::gpu_selector{});

  const int* occ_pred = _occ_pred.data_ptr<int>();
  const int* inst_xyz = _inst_xyz.data_ptr<int>();
  const int* inst_cls = _inst_cls.data_ptr<int>();
  const int* l2s_key = _l2s_key.data_ptr<int>();
  const int* inst_id_list = _inst_id_list.data_ptr<int>();
  const int* occind2detind = _occind2detind.data_ptr<int>();

  int* inst_pred = _inst_pred.data_ptr<int>();
  nearest_assign(
                 l2s_key,
                 l2s_size,
                 occind2detind,
                 inst_size,
                 occ_pred,
                 inst_xyz,
                 inst_cls,
                 inst_id_list,
                 inst_pred,
                 q
                 );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nearest_assign_forward", &nearest_assign_forward,
        "nearest_assign_forward");
}
