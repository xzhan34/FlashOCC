// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu
// SYCL implementation for Intel XPU

#include <sycl/sycl.hpp>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <cfloat>

using namespace sycl;

#define MAX_D 1446 // 700 + 700 + 45 + 1
#define MAX_STEP 1000

enum LossType {L1, L2, ABSREL};
enum PhaseName {TEST, TRAIN};

// Helper structure for int3
struct int3_sycl {
    int x, y, z;
    int3_sycl() : x(0), y(0), z(0) {}  // Default constructor
    int3_sycl(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
};

// Forward declaration of kernel functions
void init_sycl_kernel(sycl::nd_item<2> item,
                      const float* points,
                      const float* tindex,
                      float* occupancy,
                      int M, int T, int vzsize, int vysize, int vxsize);

void render_forward_sycl_kernel(sycl::nd_item<2> item,
                                const float* sigma,
                                const float* origin,
                                const float* points,
                                const float* tindex,
                                float* pred_dist,
                                float* gt_dist,
                                float* coord_index,
                                PhaseName train_phase,
                                int M, int T, int vzsize, int vysize, int vxsize);

void render_sycl_kernel(sycl::nd_item<2> item,
                        const float* sigma,
                        const float* origin,
                        const float* points,
                        const float* tindex,
                        float* pred_dist,
                        float* gt_dist,
                        float* grad_sigma,
                        LossType loss_type,
                        int M, int T, int vzsize, int vysize, int vxsize);

// Implementation - init kernel
void init_sycl_kernel(sycl::nd_item<2> item,
                      const float* points,
                      const float* tindex,
                      float* occupancy,
                      int M, int T, int vzsize, int vysize, int vxsize) {

    const auto n = item.get_group(1);  // batch index
    const auto c = item.get_global_id(0);  // ray index

    if (c < M) {
        const auto t = (int)tindex[n * M + c];

        if (t < 0) return;

        const auto ts = (T == 1) ? 0 : t;

        const int vx = (int)points[n * M * 3 + c * 3 + 0];
        const int vy = (int)points[n * M * 3 + c * 3 + 1];
        const int vz = (int)points[n * M * 3 + c * 3 + 2];

        if (0 <= vx && vx < vxsize &&
            0 <= vy && vy < vysize &&
            0 <= vz && vz < vzsize) {
            int idx = n * T * vzsize * vysize * vxsize +
                     ts * vzsize * vysize * vxsize +
                     vz * vysize * vxsize +
                     vy * vxsize +
                     vx;
            occupancy[idx] = 1;
        }
    }
}

// Implementation - render forward kernel
void render_forward_sycl_kernel(sycl::nd_item<2> item,
                                const float* sigma,
                                const float* origin,
                                const float* points,
                                const float* tindex,
                                float* pred_dist,
                                float* gt_dist,
                                float* coord_index,
                                PhaseName train_phase,
                                int M, int T, int vzsize, int vysize, int vxsize) {

    const auto n = item.get_group(1);
    const auto c = item.get_global_id(0);

    if (c >= M) return;

    const auto t = (int)tindex[n * M + c];

    if (t < 0) return;

    const auto ts = (T == 1) ? 0 : t;

    const double xo = origin[n * T * 3 + t * 3 + 0];
    const double yo = origin[n * T * 3 + t * 3 + 1];
    const double zo = origin[n * T * 3 + t * 3 + 2];

    const double xe = points[n * M * 3 + c * 3 + 0];
    const double ye = points[n * M * 3 + c * 3 + 1];
    const double ze = points[n * M * 3 + c * 3 + 2];

    int vx = (int)xo;
    int vy = (int)yo;
    int vz = (int)zo;

    const int vxe = (int)xe;
    const int vye = (int)ye;
    const int vze = (int)ze;

    const double rx = xe - xo;
    const double ry = ye - yo;
    const double rz = ze - zo;
    double gt_d = sycl::sqrt(rx * rx + ry * ry + rz * rz);

    const double dx = rx / gt_d;
    const double dy = ry / gt_d;
    const double dz = rz / gt_d;

    const int stepX = (dx >= 0) ? 1 : -1;
    const int stepY = (dy >= 0) ? 1 : -1;
    const int stepZ = (dz >= 0) ? 1 : -1;

    const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
    const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
    const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

    double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX;
    double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX;
    double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX;

    const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
    const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
    const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

    int3_sycl path[MAX_D];
    double csd[MAX_D];
    double p[MAX_D];
    double d[MAX_D];

    int step = 0;
    int count = 0;
    double last_d = 0.0;

    bool was_inside = false;
    while (true) {
        bool inside = (0 <= vx && vx < vxsize) &&
                     (0 <= vy && vy < vysize) &&
                     (0 <= vz && vz < vzsize);
        if (inside) {
            was_inside = true;
            path[count] = int3_sycl(vx, vy, vz);
        } else if (was_inside) {
            break;
        }

        double _d = 0.0;
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                _d = tMaxX;
                vx += stepX;
                tMaxX += tDeltaX;
            } else {
                _d = tMaxZ;
                vz += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                _d = tMaxY;
                vy += stepY;
                tMaxY += tDeltaY;
            } else {
                _d = tMaxZ;
                vz += stepZ;
                tMaxZ += tDeltaZ;
            }
        }

        if (inside) {
            const int3_sycl &v = path[count];
            int sigma_idx = n * T * vzsize * vysize * vxsize +
                           ts * vzsize * vysize * vxsize +
                           v.z * vysize * vxsize +
                           v.y * vxsize +
                           v.x;
            const double _sigma = sigma[sigma_idx];
            const double _delta = sycl::max(0.0, _d - last_d);
            const double sd = _sigma * _delta;

            if (count == 0) {
                csd[count] = sd;
                p[count] = 1 - sycl::exp(-sd);
            } else {
                csd[count] = csd[count-1] + sd;
                p[count] = sycl::exp(-csd[count-1]) - sycl::exp(-csd[count]);
            }

            d[count] = _d;
            count++;
        }
        last_d = _d;
        step++;

        if (step > MAX_STEP) break;
    }

    if (count > 0) {
        double exp_d = d[count-1];

        const int3_sycl &v_init = path[count-1];
        int x = v_init.x;
        int y = v_init.y;
        int z = v_init.z;

        for (int i = 0; i < count; i++) {
            const int3_sycl &v = path[i];
            int sigma_idx = n * T * vzsize * vysize * vxsize +
                           ts * vzsize * vysize * vxsize +
                           v.z * vysize * vxsize +
                           v.y * vxsize +
                           v.x;
            const double occ = sigma[sigma_idx];
            if (occ > 0.5) {
                exp_d = d[i];
                x = v.x;
                y = v.y;
                z = v.z;
                break;
            }
        }

        double p_out = sycl::exp(-csd[count-1]);
        double max_d = d[count-1];

        if (train_phase == TRAIN) {
            gt_d = sycl::min(gt_d, max_d);
        }

        pred_dist[n * M + c] = exp_d;
        gt_dist[n * M + c] = gt_d;

        coord_index[n * M * 3 + c * 3 + 0] = (double)x;
        coord_index[n * M * 3 + c * 3 + 1] = (double)y;
        coord_index[n * M * 3 + c * 3 + 2] = (double)z;
    }
}

// Implementation - render backward kernel
void render_sycl_kernel(sycl::nd_item<2> item,
                        const float* sigma,
                        const float* origin,
                        const float* points,
                        const float* tindex,
                        float* pred_dist,
                        float* gt_dist,
                        float* grad_sigma,
                        LossType loss_type,
                        int M, int T, int vzsize, int vysize, int vxsize) {

    const auto n = item.get_group(1);
    const auto c = item.get_global_id(0);

    if (c >= M) return;

    const auto t = (int)tindex[n * M + c];

    if (t < 0) return;

    const auto ts = (T == 1) ? 0 : t;

    const double xo = origin[n * T * 3 + t * 3 + 0];
    const double yo = origin[n * T * 3 + t * 3 + 1];
    const double zo = origin[n * T * 3 + t * 3 + 2];

    const double xe = points[n * M * 3 + c * 3 + 0];
    const double ye = points[n * M * 3 + c * 3 + 1];
    const double ze = points[n * M * 3 + c * 3 + 2];

    int vx = (int)xo;
    int vy = (int)yo;
    int vz = (int)zo;

    const double rx = xe - xo;
    const double ry = ye - yo;
    const double rz = ze - zo;
    double gt_d = sycl::sqrt(rx * rx + ry * ry + rz * rz);

    const double dx = rx / gt_d;
    const double dy = ry / gt_d;
    const double dz = rz / gt_d;

    const int stepX = (dx >= 0) ? 1 : -1;
    const int stepY = (dy >= 0) ? 1 : -1;
    const int stepZ = (dz >= 0) ? 1 : -1;

    const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
    const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
    const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

    double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX;
    double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX;
    double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX;

    const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
    const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
    const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

    int3_sycl path[MAX_D];
    double csd[MAX_D];
    double p[MAX_D];
    double d[MAX_D];
    double dt[MAX_D];

    int step = 0;
    int count = 0;
    double last_d = 0.0;

    bool was_inside = false;
    while (true) {
        bool inside = (0 <= vx && vx < vxsize) &&
                     (0 <= vy && vy < vysize) &&
                     (0 <= vz && vz < vzsize);
        if (inside) {
            was_inside = true;
            path[count] = int3_sycl(vx, vy, vz);
        } else if (was_inside) {
            break;
        } else if (last_d > gt_d) {
            break;
        }

        double _d = 0.0;
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                _d = tMaxX;
                vx += stepX;
                tMaxX += tDeltaX;
            } else {
                _d = tMaxZ;
                vz += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                _d = tMaxY;
                vy += stepY;
                tMaxY += tDeltaY;
            } else {
                _d = tMaxZ;
                vz += stepZ;
                tMaxZ += tDeltaZ;
            }
        }

        if (inside) {
            const int3_sycl &v = path[count];
            int sigma_idx = n * T * vzsize * vysize * vxsize +
                           ts * vzsize * vysize * vxsize +
                           v.z * vysize * vxsize +
                           v.y * vxsize +
                           v.x;
            const double _sigma = sigma[sigma_idx];
            const double _delta = sycl::max(0.0, _d - last_d);
            const double sd = _sigma * _delta;

            if (count == 0) {
                csd[count] = sd;
                p[count] = 1 - sycl::exp(-sd);
            } else {
                csd[count] = csd[count-1] + sd;
                p[count] = sycl::exp(-csd[count-1]) - sycl::exp(-csd[count]);
            }

            d[count] = _d;
            dt[count] = _delta;
            count++;
        }
        last_d = _d;
        step++;

        if (step > MAX_STEP) break;
    }

    if (count > 0) {
        double exp_d = 0.0;
        for (int i = 0; i < count; i++)
            exp_d += p[i] * d[i];

        double p_out = sycl::exp(-csd[count-1]);
        double max_d = d[count-1];

        exp_d += (p_out * max_d);
        gt_d = sycl::min(gt_d, max_d);

        pred_dist[n * M + c] = exp_d;
        gt_dist[n * M + c] = gt_d;

        // Backward pass
        double dd_dsigma[MAX_D];
        for (int i = count - 1; i >= 0; i--) {
            if (i == count - 1)
                dd_dsigma[i] = p_out * max_d;
            else
                dd_dsigma[i] = dd_dsigma[i+1] - sycl::exp(-csd[i]) * (d[i+1] - d[i]);
        }

        for (int i = count - 1; i >= 0; i--)
            dd_dsigma[i] *= dt[i];

        for (int i = count - 1; i >= 0; i--)
            dd_dsigma[i] -= dt[i] * p_out * max_d;

        double dl_dd = 1.0;
        if (loss_type == L1)
            dl_dd = (exp_d >= gt_d) ? 1 : -1;
        else if (loss_type == L2)
            dl_dd = (exp_d - gt_d);
        else if (loss_type == ABSREL)
            dl_dd = (exp_d >= gt_d) ? (1.0/gt_d) : -(1.0/gt_d);

        for (int i = 0; i < count; i++) {
            const int3_sycl &v = path[i];
            int sigma_idx = n * T * vzsize * vysize * vxsize +
                           ts * vzsize * vysize * vxsize +
                           v.z * vysize * vxsize +
                           v.y * vxsize +
                           v.x;
            // Atomic add for gradient accumulation
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                atomic_grad(grad_sigma[sigma_idx]);
            atomic_grad.fetch_add(dl_dd * dd_dsigma[i]);
        }
    }
}

// Host functions to launch kernels
void init_sycl(const float* points, const float* tindex, float* occupancy,
               int N, int M, int T, int H, int L, int W, sycl::queue& q) {

    int threads = 1024;
    sycl::range<2> global_size(((M + threads - 1) / threads) * threads, N);
    sycl::range<2> local_size(threads, 1);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
            init_sycl_kernel(item, points, tindex, occupancy, M, T, H, L, W);
        });
    }).wait();
}

void render_forward_sycl(const float* sigma, const float* origin, const float* points,
                        const float* tindex, float* pred_dist, float* gt_dist, float* coord_index,
                        int N, int M, int T, int H, int L, int W,
                        std::string phase_name, sycl::queue& q) {

    PhaseName train_phase = (phase_name == "test") ? TEST : TRAIN;

    int threads = 1024;
    sycl::range<2> global_size(((M + threads - 1) / threads) * threads, N);
    sycl::range<2> local_size(threads, 1);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
            render_forward_sycl_kernel(item, sigma, origin, points, tindex,
                                      pred_dist, gt_dist, coord_index,
                                      train_phase, M, T, H, L, W);
        });
    }).wait();
}

void render_sycl(const float* sigma, const float* origin, const float* points,
                const float* tindex, float* pred_dist, float* gt_dist, float* grad_sigma,
                int N, int M, int T, int H, int L, int W,
                std::string loss_name, sycl::queue& q) {

    LossType loss_type;
    if (loss_name == "l1") loss_type = L1;
    else if (loss_name == "l2") loss_type = L2;
    else if (loss_name == "absrel") loss_type = ABSREL;
    else loss_type = L1;

    int threads = 1024;
    sycl::range<2> global_size(((M + threads - 1) / threads) * threads, N);
    sycl::range<2> local_size(threads, 1);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
            render_sycl_kernel(item, sigma, origin, points, tindex,
                              pred_dist, gt_dist, grad_sigma,
                              loss_type, M, T, H, L, W);
        });
    }).wait();
}
