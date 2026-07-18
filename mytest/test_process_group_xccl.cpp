#include <sycl/sycl.hpp>
#include <mpi.h>
#include <oneapi/ccl.hpp>

#include <cstdlib>
#include <iostream>

static constexpr size_t N = 1024;
static int pass_count = 0;
static int fail_count = 0;

#define TEST(name, cond)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::cout << "  FAIL " << name << std::endl;                              \
      fail_count++;                                                             \
    } else {                                                                    \
      std::cout << "  PASS " << name << std::endl;                              \
      pass_count++;                                                             \
    }                                                                           \
  } while (0)

// ----------------------------------------------------------------
void test_comm_rank_size(int rank, int size,
                         ccl::communicator &comm) {
  TEST("comm rank", comm.rank() == rank);
  TEST("comm size", comm.size() == size);
}

// ----------------------------------------------------------------
void test_trivial_allreduce(int rank, int size,
                            ccl::communicator &comm,
                            ccl::stream &stream,
                            sycl::queue &q) {
  auto buf = sycl::malloc_device<int>(1, q);
  q.memcpy(buf, &rank, sizeof(int)).wait();
  // in-place allreduce: rank+1
  int init = rank + 1;
  q.memcpy(buf, &init, sizeof(int)).wait();
  ccl::allreduce(buf, buf, 1, ccl::datatype::int32, ccl::reduction::sum,
                 comm, stream).wait();
  int result;
  q.memcpy(&result, buf, sizeof(int)).wait();
  sycl::free(buf, q);
  int expected = size * (size + 1) / 2;
  TEST("trivial allreduce", result == expected);
}

// ----------------------------------------------------------------
void test_non_float_types(int rank, int size,
                          ccl::communicator &comm,
                          ccl::stream &stream,
                          sycl::queue &q) {
  // int32
  {
    auto buf = sycl::malloc_device<int>(N, q);
    q.parallel_for(N, [=](auto id) { buf[id] = rank + 1; }).wait();
    ccl::allreduce(buf, buf, N, ccl::datatype::int32, ccl::reduction::sum,
                   comm, stream).wait();
    int *host = new int[N];
    q.memcpy(host, buf, N * sizeof(int)).wait();
    int expected = size * (size + 1) / 2;
    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i) ok = (host[i] == expected);
    TEST("int32 allreduce", ok);
    delete[] host;
    sycl::free(buf, q);
  }

  // int64
  {
    auto buf = sycl::malloc_device<int64_t>(N, q);
    q.parallel_for(N, [=](auto id) { buf[id] = (int64_t)(rank + 1); }).wait();
    ccl::allreduce(buf, buf, N, ccl::datatype::int64, ccl::reduction::sum,
                   comm, stream).wait();
    auto *host = new int64_t[N];
    q.memcpy(host, buf, N * sizeof(int64_t)).wait();
    int64_t expected = size * (size + 1) / 2;
    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i) ok = (host[i] == expected);
    TEST("int64 allreduce", ok);
    delete[] host;
    sycl::free(buf, q);
  }

  // uint8
  {
    auto buf = sycl::malloc_device<uint8_t>(N, q);
    q.parallel_for(N, [=](auto id) { buf[id] = (uint8_t)(rank + 1); }).wait();
    ccl::allreduce(buf, buf, N, ccl::datatype::uint8, ccl::reduction::sum,
                   comm, stream).wait();
    auto *host = new uint8_t[N];
    q.memcpy(host, buf, N * sizeof(uint8_t)).wait();
    uint8_t expected = size * (size + 1) / 2;
    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i) ok = (host[i] == expected);
    TEST("uint8 allreduce", ok);
    delete[] host;
    sycl::free(buf, q);
  }
}

// ----------------------------------------------------------------
void test_empty_allreduce(ccl::communicator &comm,
                          ccl::stream &stream,
                          sycl::queue &q) {
  auto buf = sycl::malloc_device<int>(0, q);
  try {
    ccl::allreduce(buf, buf, 0, ccl::datatype::int32, ccl::reduction::sum,
                   comm, stream).wait();
    TEST("empty allreduce (no-op)", true);
  } catch (...) {
    TEST("empty allreduce (no-op)", false);
  }
  sycl::free(buf, q);
}

// ----------------------------------------------------------------
void test_comm_split(int rank, int size,
                     ccl::communicator &comm,
                     ccl::stream &stream,
                     sycl::queue &q) {
  int color = rank % 2;
  auto sub = ccl::split_communicator(comm, color, rank);
  int expected_sub_size = (size + (1 - color)) / 2;
  int expected_sub_rank = rank / 2;
  TEST("split comm size", sub.size() == expected_sub_size);
  TEST("split comm rank", sub.rank() == expected_sub_rank);

  auto buf = sycl::malloc_device<int>(1, q);
  int init = rank + 1;
  q.memcpy(buf, &init, sizeof(int)).wait();
  ccl::allreduce(buf, buf, 1, ccl::datatype::int32, ccl::reduction::sum,
                 sub, stream).wait();
  int result;
  q.memcpy(&result, buf, sizeof(int)).wait();
  sycl::free(buf, q);

  int expected = 0;
  for (int r = 0; r < size; ++r)
    if (r % 2 == color) expected += (r + 1);
  TEST("split allreduce", result == expected);
}

// ----------------------------------------------------------------
void test_multi_comm_cleanup(int rank, int size,
                             ccl::communicator &comm,
                             ccl::stream &stream,
                             sycl::queue &q) {
  auto buf = sycl::malloc_device<int>(N, q);
  q.parallel_for(N, [=](auto id) { buf[id] = rank + 1; }).wait();

  auto sub1 = ccl::split_communicator(comm, 0, rank);
  auto sub2 = ccl::split_communicator(comm, 0, rank);

  ccl::allreduce(buf, buf, N, ccl::datatype::int32, ccl::reduction::sum,
                 sub1, stream).wait();
  ccl::allreduce(buf, buf, N, ccl::datatype::int32, ccl::reduction::sum,
                 sub2, stream).wait();

  int expected = (size * (size + 1) / 2) * 2;
  auto *host = new int[N];
  q.memcpy(host, buf, N * sizeof(int)).wait();
  bool ok = true;
  for (size_t i = 0; i < N && ok; ++i) ok = (host[i] == expected);
  TEST("multi comm allreduce", ok);
  delete[] host;
  sycl::free(buf, q);
}

// ----------------------------------------------------------------
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ccl::init();

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (rank >= (int)devices.size()) {
    std::cerr << "[rank " << rank << "] no GPU device" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  sycl::queue q(devices[rank], sycl::property::queue::in_order{});

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

  if (rank == 0) std::cout << "--- test_comm_rank_size ---" << std::endl;
  test_comm_rank_size(rank, size, comm);

  if (rank == 0) std::cout << "--- test_trivial_allreduce ---" << std::endl;
  test_trivial_allreduce(rank, size, comm, stream, q);

  if (rank == 0) std::cout << "--- test_non_float_types ---" << std::endl;
  test_non_float_types(rank, size, comm, stream, q);

  if (rank == 0) std::cout << "--- test_empty_allreduce ---" << std::endl;
  test_empty_allreduce(comm, stream, q);

  if (rank == 0) std::cout << "--- test_comm_split ---" << std::endl;
  test_comm_split(rank, size, comm, stream, q);

  if (rank == 0) std::cout << "--- test_multi_comm_cleanup ---" << std::endl;
  test_multi_comm_cleanup(rank, size, comm, stream, q);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\n" << pass_count << " passed, " << fail_count
              << " failed" << std::endl;
  }

  MPI_Finalize();
  return fail_count > 0 ? 1 : 0;
}
