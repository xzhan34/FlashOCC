// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu
// Modified for Intel XPU with SYCL

#include <string>
#include <torch/extension.h>
#include <vector>
#include <sycl/sycl.hpp>

/*
 * SYCL forward declarations
 */

void render_forward_sycl(const float* sigma, const float* origin, const float* points,
                        const float* tindex, float* pred_dist, float* gt_dist, float* coord_index,
                        int N, int M, int T, int H, int L, int W,
                        std::string phase_name, sycl::queue& q);

void render_sycl(const float* sigma, const float* origin, const float* points,
                const float* tindex, float* pred_dist, float* gt_dist, float* grad_sigma,
                int N, int M, int T, int H, int L, int W,
                std::string loss_name, sycl::queue& q);

void init_sycl(const float* points, const float* tindex, float* occupancy,
               int N, int M, int T, int H, int L, int W, sycl::queue& q);


/*
 * C++ interface
 */

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be an XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_XPU(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
render_forward(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
               torch::Tensor tindex, const std::vector<int> grid,
               std::string phase_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);

  const auto N = points.size(0);
  const auto M = points.size(1);
  const auto T = grid[0];
  const auto H = grid[1];
  const auto L = grid[2];
  const auto W = grid[3];

  const auto device = sigma.device();
  sycl::queue q = sycl::queue(sycl::gpu_selector{});

  auto gt_dist = -torch::ones({N, M}, device);
  auto pred_dist = -torch::ones({N, M}, device);
  auto coord_index = torch::zeros({N, M, 3}, device);

  render_forward_sycl(
    sigma.data_ptr<float>(),
    origin.data_ptr<float>(),
    points.data_ptr<float>(),
    tindex.data_ptr<float>(),
    pred_dist.data_ptr<float>(),
    gt_dist.data_ptr<float>(),
    coord_index.data_ptr<float>(),
    N, M, T, H, L, W, phase_name, q
  );

  return {pred_dist, gt_dist, coord_index};
}


std::vector<torch::Tensor> render(torch::Tensor sigma, torch::Tensor origin,
                                  torch::Tensor points, torch::Tensor tindex,
                                  std::string loss_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);

  const auto N = points.size(0);
  const auto M = points.size(1);
  const auto T = sigma.size(1);
  const auto H = sigma.size(2);
  const auto L = sigma.size(3);
  const auto W = sigma.size(4);

  const auto device = sigma.device();
  sycl::queue q = sycl::queue(sycl::gpu_selector{});

  auto gt_dist = -torch::ones({N, M}, device);
  auto pred_dist = -torch::ones({N, M}, device);
  auto grad_sigma = torch::zeros_like(sigma);

  render_sycl(
    sigma.data_ptr<float>(),
    origin.data_ptr<float>(),
    points.data_ptr<float>(),
    tindex.data_ptr<float>(),
    pred_dist.data_ptr<float>(),
    gt_dist.data_ptr<float>(),
    grad_sigma.data_ptr<float>(),
    N, M, T, H, L, W, loss_name, q
  );

  return {pred_dist, gt_dist, grad_sigma};
}

torch::Tensor init(torch::Tensor points, torch::Tensor tindex,
                   const std::vector<int> grid) {
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);

  const auto N = points.size(0);
  const auto M = points.size(1);
  const auto T = grid[0];
  const auto H = grid[1];
  const auto L = grid[2];
  const auto W = grid[3];

  const auto dtype = points.dtype();
  const auto device = points.device();
  const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
  auto occupancy = torch::zeros({N, T, H, L, W}, options);

  sycl::queue q = sycl::queue(sycl::gpu_selector{});

  init_sycl(
    points.data_ptr<float>(),
    tindex.data_ptr<float>(),
    occupancy.data_ptr<float>(),
    N, M, T, H, L, W, q
  );

  return occupancy;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "Initialize");
  m.def("render", &render, "Render");
  m.def("render_forward", &render_forward, "Render (forward pass only)");
}
