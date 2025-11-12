#ifndef _BEV_SUM_POOL_H
#define _BEV_SUM_POOL_H

#include <torch/torch.h>
#include <ATen/core/TensorBody.h>
#include <sycl/sycl.hpp>

at::Tensor bev_sum_pool_forward(
  const at::Tensor _geom_feats,
  const at::Tensor _geom_coords,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
);

at::Tensor bev_sum_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_coords,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
);


// CUDA function declarations
void bev_sum_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out,
    sycl::queue& q);

void bev_sum_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad,
  sycl::queue& q);


#endif