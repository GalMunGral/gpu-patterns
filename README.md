#   CUDA Samples

A collection of CUDA programs illustrating common GPU parallelism patterns.

| File | Pattern |
|------|---------|
| `vec_add.cu` | Basic data parallelism |
| `gemm_simple.cu` | Dense matrix multiplication |
| `gemm_tiling.cu` | Tiled matrix multiplication with shared memory |
| `histogram.cu` | Atomic operations |
| `reduction.cu` | Parallel reduction |
| `scan_brent_kung.cu` | Prefix scan (Brent-Kung algorithm) |
| `spmv_jds.cu` | Sparse matrix-vector multiplication (JDS format) |
| `conv_3d.cu` | 3D convolution |
| `cnn-forward.cu` | CNN forward pass |

## Rhetorical Design

### Purpose

Modern deep learning is powered by GPUs, but the connection between a neural network and the hardware executing it is rarely made explicit. This collection demystifies that connection. The CNN forward pass — the core computation of a convolutional neural network — reduces to matrix multiplication, which is itself just another instance of the GPU parallelism patterns illustrated by the other programs in this collection. There is no special machinery: the same thread/block model, shared memory, and memory access discipline that governs vector addition and reduction also governs neural network inference.

### Strategy

The programs are ordered roughly by complexity. The simpler kernels — vector addition and naive matrix multiply — establish the thread/block model. Tiled matrix multiply introduces shared memory to exploit data reuse. Reduction and prefix scan illustrate tree-structured parallelism. Sparse matrix-vector multiply shows how data layout affects parallelism. The CNN forward pass demonstrates how convolution can be transformed into matrix multiplication, reducing a new problem to one already solved efficiently on the GPU.