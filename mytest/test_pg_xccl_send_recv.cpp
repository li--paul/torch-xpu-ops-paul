// Demonstrate ProcessGroupXCCL::send/recv from C++.
//
// Build:
//   icpx -fsycl -DUSE_C10D_XCCL -Doverride= \
//     -I/path/to/torch-xpu-ops/src \
//     -I/path/to/torch/include \
//     -I/path/to/torch/include/torch/csrc/api/include \
//     -D_GLIBCXX_USE_CXX11_ABI=1 \
//     test_pg_xccl_send_recv.cpp \
//     -L/path/to/torch/lib \
//     -ltorch_xpu -ltorch_cpu -ltorch_python -ltorch -lc10_xpu -lc10 \
//     -o test_pg_xccl_send_recv
//
// Run:
//   export LD_LIBRARY_PATH=/path/to/torch/lib
//   mpirun -np 2 ./test_pg_xccl_send_recv <rank> <world_size>
//   # or manually in two terminals:
//   ./test_pg_xccl_send_recv 0 2   # terminal 1
//   ./test_pg_xccl_send_recv 1 2   # terminal 2

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <xccl/ProcessGroupXCCL.hpp>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/impl/XPUGuardImpl.h>

#include <iostream>
#include <memory>
#include <string>

using namespace c10d;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <rank> <world_size>\n";
    return 1;
  }
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  // === 1. Create Store and ProcessGroupXCCL ===
  auto store = c10::make_intrusive<FileStore>("/tmp/pg_demo", world_size);

  auto pg = c10::make_intrusive<ProcessGroupXCCL>(store, rank, world_size);

  // === 2. Create XPU tensor ===
  auto device = torch::Device(torch::kXPU, rank);
  torch::Tensor tensor;
  if (rank == 0) {
    tensor = torch::full({10, 5}, 42.0f,
                         torch::TensorOptions().device(device));
    std::cout << "[rank " << rank << "] sending tensor of shape "
              << tensor.sizes() << std::endl;
  } else {
    tensor = torch::empty({10, 5}, torch::TensorOptions().device(device));
  }

  // === 3. send / recv via ProcessGroupXCCL ===
  //     pg->send({tensor}, dstRank, tag)->wait()
  //     pg->recv({tensor}, srcRank, tag)->wait()
  // send/recv are protected in ProcessGroupXCCL but public in Backend.
  // Cast to Backend* to call through the base class interface.
  auto* backend = static_cast<c10d::Backend*>(pg.get());

  std::vector<at::Tensor> tensors = {tensor};
  c10::intrusive_ptr<Work> work;

  if (rank == 0) {
    work = backend->send(tensors, /*dstRank=*/1, /*tag=*/0);
  } else {
    work = backend->recv(tensors, /*srcRank=*/0, /*tag=*/0);
  }
  work->wait();
  c10::xpu::getCurrentXPUStream(device.index()).queue().wait();

  // === 4. Verify ===
  if (rank == 1) {
    auto expected = torch::full({10, 5}, 42.0f,
                                torch::TensorOptions().device(device));
    if (tensor.equal(expected)) {
      std::cout << "[rank " << rank << "] PASS: received tensor matches\n";
    } else {
      std::cout << "[rank " << rank << "] FAIL: data mismatch\n";
      return 1;
    }
  } else {
    std::cout << "[rank " << rank << "] PASS: send completed\n";
  }

  return 0;
}
