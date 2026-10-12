CXX      := g++
MPICXX   := mpic++
CXXFLAGS := -O2 -Wall -std=c++17 -Iinclude -Isrc/common
OMPFLAGS := -fopenmp

BIN := bin
SEQ_BIN    := $(BIN)/dna_seq
OMP_BIN    := $(BIN)/dna_openmp
MPI_BIN    := $(BIN)/dna_mpi
HYBRID_BIN := $(BIN)/dna_hybrid

.PHONY: all clean test dirs sequential openmp mpi hybrid

all: dirs sequential openmp mpi hybrid

dirs:
	@mkdir -p $(BIN)

sequential: dirs
	$(CXX) $(CXXFLAGS) src/sequential/main.cpp -o $(SEQ_BIN)

openmp: dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) src/openmp/main.cpp -o $(OMP_BIN)

mpi: dirs
	$(MPICXX) $(CXXFLAGS) src/mpi/main.cpp -o $(MPI_BIN)

hybrid: dirs
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) src/hybrid/main.cpp -o $(HYBRID_BIN)

# --- Tests (correctness validation, Spec section 21) ---
TEST_BIN := $(BIN)/run_tests

test: dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) tests/test_all.cpp -o $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BIN)
	rm -rf results/raw/*.csv results/processed/*.csv graphs/*.png
