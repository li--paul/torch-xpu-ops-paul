#include <torch/extension.h>
#include <torch/types.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

// Demo op 1: get device index + queue pointer from tensor
std::string whoami(const at::Tensor& input) {
  auto device = input.device();
  if (!device.is_xpu()) {
    return "not an XPU tensor";
  }
  int dev_idx = device.index();
  auto& queue = c10::xpu::getCurrentXPUStream(dev_idx).queue();
  std::ostringstream oss;
  oss << "device=xpu:" << dev_idx
      << ", sycl::queue=" << &queue
      << ", backend=" << (queue.get_backend() == sycl::backend::ext_oneapi_level_zero ? "Level Zero" : "other");
  return oss.str();
}

// Demo op 2: add one using SYCL kernel on the tensor's device
at::Tensor add_one(const at::Tensor& input) {
  TORCH_CHECK(input.is_xpu(), "input must be an XPU tensor");
  int dev_idx = input.device().index();
  auto& queue = c10::xpu::getCurrentXPUStream(dev_idx).queue();
  auto output = at::empty_like(input);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "add_one_xpu", [&] {
    auto n = input.numel();
    const scalar_t* in_ptr = input.data_ptr<scalar_t>();
    scalar_t* out_ptr = output.data_ptr<scalar_t>();

    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        out_ptr[i] = in_ptr[i] + static_cast<scalar_t>(1);
      });
    });
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("whoami", &whoami, "Show XPU device and SYCL queue info");
  m.def("add_one", &add_one, "Add 1 to every element via SYCL kernel");
}
