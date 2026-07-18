#include <sycl/sycl.hpp>
#include <mpi.h>
#include <oneapi/ccl.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size != 2) {
    std::cerr << "requires exactly 2 ranks" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ccl::init();

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  sycl::queue q(devices[rank], sycl::property::queue::in_order{});

  // KVS rendezvous
  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type addr;
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    addr = kvs->get_address();
    MPI_Bcast((void*)addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast((void*)addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    kvs = ccl::create_kvs(addr);
  }

  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
  auto stream = ccl::create_stream(q);

  // Each rank has a send_buf and a recv_buf on device
  size_t n = 10;
  int* send_buf = sycl::malloc_device<int>(n, q);
  int* recv_buf = sycl::malloc_device<int>(n, q);

  // Init send_buf with rank-specific values
  q.parallel_for(n, [=](auto id) {
     send_buf[id] = rank * 100 + (int)id;
     recv_buf[id] = -1;
   }).wait();

  std::cout << "[rank " << rank << "] before: send_buf[0]="
            << (rank * 100) << ", recv_buf[0]=" << -1 << std::endl;

  // ── send / recv ────────────────────────────────────────────
  // ProcessGroupXCCL internally calls onecclSend/onecclRecv
  // exactly like this, wrapped in groupStart/groupEnd for batching.
  ccl::group_start();
  if (rank == 0) {
    // rank 0 sends to rank 1, then receives from rank 1
    ccl::send(send_buf, n, ccl::datatype::int32, 1, comm, stream);
    ccl::recv(recv_buf, n, ccl::datatype::int32, 1, comm, stream);
  } else {
    // rank 1 receives from rank 0, then sends to rank 0
    ccl::recv(recv_buf, n, ccl::datatype::int32, 0, comm, stream);
    ccl::send(send_buf, n, ccl::datatype::int32, 0, comm, stream);
  }
  ccl::group_end();

  q.wait();

  // Verify
  int h_send[10], h_recv[10];
  q.memcpy(h_send, send_buf, n * sizeof(int)).wait();
  q.memcpy(h_recv, recv_buf, n * sizeof(int)).wait();

  bool ok = true;
  for (size_t i = 0; i < n; ++i) {
    // send_buf should be unchanged
    if (h_send[i] != rank * 100 + (int)i) ok = false;
    // recv_buf should contain peer's data
    int peer = (rank == 0) ? 1 : 0;
    if (h_recv[i] != peer * 100 + (int)i) {
      std::cout << "[rank " << rank << "] FAIL at [" << i
                << "]: recv=" << h_recv[i]
                << " expected=" << (peer * 100 + (int)i) << std::endl;
      ok = false;
    }
  }

  if (ok) {
    std::cout << "[rank " << rank << "] PASS: recv_buf[0]="
              << h_recv[0] << " (peer's send_buf[0])" << std::endl;
  }

  sycl::free(send_buf, q);
  sycl::free(recv_buf, q);
  MPI_Finalize();
  return ok ? 0 : 1;
}
