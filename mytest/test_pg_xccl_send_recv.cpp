/*

  ProcessGroupXCCL::send / recv 演示

  ProcessGroupXCCL.hpp 无法用 icpx 在项目外编译（头文件依赖链中
  namespace c10d 的可见性被 icpx 破坏）。实际 ProcessGroupXCCL
  已经编译在 libtorch_xpu.so 中，通过 Python torch.distributed
  接口即可调用。

  C++ 等价的调用链:

    // 1. 创建 Store + ProcessGroupXCCL
    auto store = c10::make_intrusive<FileStore>("/tmp/demo", world_size);
    auto pg = c10::make_intrusive<ProcessGroupXCCL>(store, rank, size);

    // 2. send / recv
    std::vector<at::Tensor> tensors = {tensor};
    c10::intrusive_ptr<Work> work;
    if (rank == 0)
      work = pg->send(tensors, dstRank=1, tag=0);
    else
      work = pg->recv(tensors, srcRank=0, tag=0);
    work->wait();

    // 3. 清理
    dist->destroy_process_group();

  底层：pg->send() → pointToPoint() → xccl::onecclSend() → onecclSend()
        pg->recv() → pointToPoint() → xccl::onecclRecv() → onecclRecv()

  纯 C++ 版直接用 oneCCL API 的演示见 test_single_p2p.cpp。

*/
int main() { return 0; }
