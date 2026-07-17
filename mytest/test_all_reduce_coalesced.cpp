#include <sycl/sycl.hpp>
#include <mpi.h>
#include <oneapi/ccl.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static constexpr int NTENSORS = 5;
static constexpr int BASE_SIZE = 60;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size != 2) {
    std::cerr << "this test requires exactly 2 ranks" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ccl::init();

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (rank >= (int)devices.size()) {
    std::cerr << "rank " << rank << " has no GPU device" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  sycl::queue q(devices[rank], sycl::property::queue::in_order{});

  // KVS rendezvous
  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type addr;
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    addr = kvs->get_address();
    MPI_Bcast((void *)addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast((void *)addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    kvs = ccl::create_kvs(addr);
  }

  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
  auto stream = ccl::create_stream(q);

  // Allocate 5 USM device buffers of sizes [60, 61, 62, 63, 64]
  int *bufs[NTENSORS];
  size_t sizes[NTENSORS];
  for (int i = 0; i < NTENSORS; ++i) {
    sizes[i] = BASE_SIZE + i;
    bufs[i] = sycl::malloc_device<int>(sizes[i], q);
    int init_val = rank + 1 + i;
    int *buf = bufs[i];
    q.parallel_for(sizes[i], [=](auto id) {
       buf[id] = init_val;
     }).wait();
  }

  // === test all_reduce_coalesced: group_start + allreduce + group_end ===
  ccl::group_start();
  for (int i = 0; i < NTENSORS; ++i) {
    ccl::allreduce(bufs[i], bufs[i], sizes[i], ccl::datatype::int32,
                   ccl::reduction::sum, comm, stream);
  }
  ccl::group_end();

  bool pass = true;
  for (int i = 0; i < NTENSORS; ++i) {
    float expected_f = size * (i + (size + 1.0f) / 2.0f);
    int expected = (int)expected_f;

    int *host = new int[sizes[i]];
    q.memcpy(host, bufs[i], sizes[i] * sizeof(int)).wait();

    for (size_t j = 0; j < sizes[i]; ++j) {
      if (host[j] != expected) {
        pass = false;
        std::cout << "[rank " << rank << "] FAIL buf[" << i << "][" << j
                  << "]: got " << host[j] << ", expected " << expected
                  << std::endl;
      }
    }
    delete[] host;
  }

  if (pass)
    std::cout << "[rank " << rank << "] PASS coalesced allreduce" << std::endl;

  for (int i = 0; i < NTENSORS; ++i)
    sycl::free(bufs[i], q);

  MPI_Finalize();
  return pass ? 0 : 1;
}
