# Makefile - build every .cu into an executable with the same base name (no extension)

NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++14
LDFLAGS ?=

SRCS := $(wildcard *.cu)
EXES := $(SRCS:.cu=)

.PHONY: all clean help
all: $(EXES)
ifeq ($(strip $(EXES)),)
	@echo "No .cu files found."
endif

# Pattern rule: build executable (no extension) from .cu source
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	-@rm -f $(EXES) *.o *~ a.out

help:
	@echo "Usage:"
	@echo "  make        Build all CUDA executables (produces files with no extension)"
	@echo "  make exe    Build specific executable 'exe' from 'exe.cu'"
	@echo "  make -jN    Parallel build"
	@echo "  make clean  Remove built executables"
