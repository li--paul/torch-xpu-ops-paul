#include <sycl/sycl.hpp>
#include <mpi.h>
#include <oneapi/ccl.hpp>

#include <cstdlib>
#include <iostream>

static constexpr size_t N = 10;

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

  // Create SYCL queue on xpu:rank
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

  // Create oneCCL communicator
  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
  auto stream = ccl::create_stream(q);

  // Allocate USM device memory
  int *send_buf = sycl::malloc_device<int>(N, q);
  int *recv_buf = sycl::malloc_device<int>(N, q);

  // Init send_buf on device, recv_buf to zero
  q.parallel_for(N, [=](auto id) {
     send_buf[id] = rank + (int)id;
     recv_buf[id] = 0;
   }).wait();

  std::cout << "[rank " << rank << "] sending from GPU "
            << devices[rank].get_info<sycl::info::device::name>() << std::endl;

  if (rank == 0) {
    ccl::send(send_buf, N, ccl::datatype::int32, 1, comm, stream).wait();
    ccl::recv(recv_buf, N, ccl::datatype::int32, 1, comm, stream).wait();
  } else {
    ccl::recv(recv_buf, N, ccl::datatype::int32, 0, comm, stream).wait();
    ccl::send(send_buf, N, ccl::datatype::int32, 0, comm, stream).wait();
  }

  // Copy recv_buf back to host for verification
  int *host_recv = new int[N];
  int *host_send = new int[N];
  q.memcpy(host_recv, recv_buf, N * sizeof(int)).wait();
  q.memcpy(host_send, send_buf, N * sizeof(int)).wait();

  bool pass = true;
  for (size_t i = 0; i < N; ++i) {
    int expected;
    if (rank == 0)
      expected = 1 + i; // rank 1's send_buf
    else
      expected = 0 + i; // rank 0's send_buf
    if (host_recv[i] != expected) {
      pass = false;
      std::cout << "[rank " << rank << "] FAIL at [" << i
                << "]: got " << host_recv[i] << ", expected " << expected
                << std::endl;
    }
  }

  if (pass)
    std::cout << "[rank " << rank << "] PASS" << std::endl;

  // Cleanup
  sycl::free(send_buf, q);
  sycl::free(recv_buf, q);
  delete[] host_recv;
  delete[] host_send;

  MPI_Finalize();
  return pass ? 0 : 1;
}
