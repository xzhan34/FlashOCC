// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
// SYCL implementation for Intel XPU

#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace sycl;

void nearest_assign_kernel(sycl::nd_item<1> item,
                           const int* l2s_key,
                           int l2s_size,
                           const int* occind2detind,
                           const int *occ_pred,
                           const int *inst_xyz,
                           const int *inst_cls,
                           const int *inst_id_list,
                           int inst_size,
                           int* inst_pred) {
  int idx = item.get_global_id(0);

  if (idx < 200*200*16)
  {
    int occ_pred_label = occ_pred[idx];
    int dist_min = 100000000;
    for (int index = 0; index < l2s_size; index ++)
    {
      if (occ_pred_label == l2s_key[index])
      {
        int x = idx/(200*16);
        int y = (idx - x*200*16)/16;
        int z = idx - x*200*16 - y*16;
        int inst_ind = 0;
        for (inst_ind = 0; inst_ind < inst_size; inst_ind ++)
        {
          if (inst_cls[inst_ind] == occind2detind[occ_pred_label])
          {
            int dx = x - inst_xyz[inst_ind*3+0];
            int dy = y - inst_xyz[inst_ind*3+1];
            int dz = z - inst_xyz[inst_ind*3+2];
            int dist = dx*dx + dy*dy + dz*dz;
            if (dist < dist_min){
              dist_min = dist;
              inst_pred[idx] = inst_id_list[inst_ind];
            }
          }
        }
        return;
      }
    }
    inst_pred[idx] = occ_pred[idx];
  }
}

void nearest_assign(const int* l2s_key,
                   int l2s_size,
                   const int *occind2detind,
                   int inst_size,
                   const int *occ_pred,
                   const int *inst_xyz,
                   const int *inst_cls,
                   const int *inst_id_list,
                   int* inst_pred,
                   sycl::queue& q) {

  int total_threads = 200 * 200 * 16;
  int work_group_size = 256;
  int num_work_groups = (total_threads + work_group_size - 1) / work_group_size;

  sycl::range<1> global_size(num_work_groups * work_group_size);
  sycl::range<1> local_size(work_group_size);

  q.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
    nearest_assign_kernel(item, l2s_key, l2s_size, occind2detind,
                         occ_pred, inst_xyz, inst_cls,
                         inst_id_list, inst_size, inst_pred);
  }).wait();
}
