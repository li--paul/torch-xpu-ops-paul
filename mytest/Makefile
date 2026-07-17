CCL_ROOT = /opt/intel/oneapi/ccl/2022.0
MPI_ROOT = /opt/intel/oneapi/mpi/2021.18

CXX = icpx
CXXFLAGS = -fsycl -I$(CCL_ROOT)/include -I$(MPI_ROOT)/include
LDFLAGS  = -L$(CCL_ROOT)/lib -L$(MPI_ROOT)/lib
LDLIBS   = -lccl -lmpi

test_single_p2p: test_single_p2p.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f test_single_p2p
