// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
// SYCL implementation for Intel XPU

#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace sycl;

/*
  Function: pillar pooling (SYCL kernel)
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
void bev_pool_v2_kernel(sycl::nd_item<1> item, int c, int n_intervals,
                        const float *depth,
                        const float *feat,
                        const int *ranks_depth,
                        const int *ranks_feat,
                        const int *ranks_bev,
                        const int *interval_starts,
                        const int *interval_lengths,
                        float* out) {
  int idx = item.get_global_id(0);
  int index = idx / c;  // pillar id
  int cur_c = idx % c;  // channel id
  if (index >= n_intervals) return;

  int interval_start = interval_starts[index];      // 该pillar的起始索引.
  int interval_length = interval_lengths[index];    // 该pillar的包含的点数量.
  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;

  for(int i = 0; i < interval_length; i++){
    // ranks_depth[interval_start+i]: depth索引, 介于(0, B*N*D*fH*fW-1)之间.
    cur_depth = depth + ranks_depth[interval_start+i];
    // ranks_feat[interval_start+i]: feature索引, 介于(0, B*N*fH*fW-1)之间.
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += *cur_feat * *cur_depth;  // 聚合该pillar对应的cur_c特征.
  }

  const int* cur_rank = ranks_bev + interval_start;     // 该pillar在BEV grids中对应的索引.
  float* cur_out = out + *cur_rank * c + cur_c;     // 该cur_c特征对应的索引位置.
  *cur_out = psum;
}


/*
  Function: pillar pooling backward (SYCL kernel)
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n]
    ranks_feat       : input index of feat, IntTensor[n]
    ranks_bev        : output index, IntTensor[n]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    depth_grad       : gradient of the depth fmap, FloatTensor
    feat_grad        : gradient of the feature fmap, FloatTensor
*/
void bev_pool_grad_kernel(sycl::nd_item<1> item, int c, int n_intervals,
                          const float *out_grad,
                          const float *depth,
                          const float *feat,
                          const int *ranks_depth,
                          const int *ranks_feat,
                          const int *ranks_bev,
                          const int *interval_starts,
                          const int *interval_lengths,
                          float* depth_grad,
                          float* feat_grad) {
  int idx = item.get_global_id(0);
  if (idx >= n_intervals) return;

  int interval_start = interval_starts[idx];        // 该pillar的起始索引.
  int interval_length = interval_lengths[idx];      // 该pillar的包含的点数量.

  const int* cur_rank;
  const float* cur_out_grad;
  const float* cur_out_grad_start;

  const float* cur_feat;
  const float* cur_feat_start;
  float* cur_depth_grad;
  float grad_sum;

  for(int i = 0; i < interval_length; i++){
    cur_rank = ranks_bev + interval_start + i;      // 该pillar在BEV grids中对应的索引.
    cur_out_grad_start = out_grad + *cur_rank * c;    // pillar feature 的 grad.
    cur_feat_start = feat + ranks_feat[interval_start+i] * c;

    grad_sum = 0;
    for(int cur_c = 0; cur_c < c; cur_c++){
      cur_out_grad = cur_out_grad_start + cur_c;
      cur_feat = cur_feat_start + cur_c;
      grad_sum += *cur_out_grad * *cur_feat;
    }

    cur_depth_grad = depth_grad + ranks_depth[interval_start+i];
    *cur_depth_grad = grad_sum;
  }

  float* cur_feat_grad;
  const float* cur_depth;
  for(int cur_c = 0; cur_c < c; cur_c++){
    grad_sum = 0;
    for(int i = 0; i < interval_length; i++){
      cur_rank = ranks_bev + interval_start + i;
      cur_out_grad = out_grad + *cur_rank * c + cur_c;

      cur_depth = depth + ranks_depth[interval_start+i];
      grad_sum += *cur_out_grad * *cur_depth;
    }
    cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c;
    *cur_feat_grad = grad_sum;
  }
}


void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
                 const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
                 const int* interval_starts, const int* interval_lengths, float* out,
                 sycl::queue& q) {

  int total_threads = n_intervals * c;
  int work_group_size = 256;
  int num_work_groups = (total_threads + work_group_size - 1) / work_group_size;

  sycl::range<1> global_size(num_work_groups * work_group_size);
  sycl::range<1> local_size(work_group_size);

  q.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
    bev_pool_v2_kernel(item, c, n_intervals, depth, feat, ranks_depth, ranks_feat,
                       ranks_bev, interval_starts, interval_lengths, out);
  }).wait();
}

void bev_pool_v2_grad(int c, int n_intervals, const float* out_grad,
                      const float* depth, const float* feat, const int* ranks_depth,
                      const int* ranks_feat, const int* ranks_bev,
                      const int* interval_starts, const int* interval_lengths,
                      float* depth_grad, float* feat_grad,
                      sycl::queue& q) {

  int work_group_size = 256;
  int num_work_groups = (n_intervals + work_group_size - 1) / work_group_size;

  sycl::range<1> global_size(num_work_groups * work_group_size);
  sycl::range<1> local_size(work_group_size);

  q.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
    bev_pool_grad_kernel(item, c, n_intervals, out_grad, depth, feat, ranks_depth,
                         ranks_feat, ranks_bev, interval_starts, interval_lengths,
                         depth_grad, feat_grad);
  }).wait();
}
