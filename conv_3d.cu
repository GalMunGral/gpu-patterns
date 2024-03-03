#include <wb.h>
#include <iostream>

#define wbCheck(stmt)                                              \
    do                                                             \
    {                                                              \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
        }                                                          \
    } while (0)

//@@ Define any useful program-wide constants here
constexpr int TILE_SIZE = 8;
constexpr int MASK_SIZE = 3;
constexpr int INPUT_TILE_SIZE = TILE_SIZE + MASK_SIZE - 1;

//@@ Define constant memory for device kernel here
__constant__ float K[MASK_SIZE][MASK_SIZE][MASK_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size)
{
    //@@ Insert kernel code here
    __shared__ float M[INPUT_TILE_SIZE][INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int x_o = bx * TILE_SIZE + tx, y_o = by * TILE_SIZE + ty, z_o = bz * TILE_SIZE + tz;
    int x_i = x_o - MASK_SIZE / 2, y_i = y_o - MASK_SIZE / 2, z_i = z_o - MASK_SIZE / 2;

    if (z_i >= 0 && z_i < z_size && y_i >= 0 && y_i < y_size && x_i >= 0 && x_i < x_size)
    {
        M[tz][ty][tx] = input[z_i * y_size * x_size + y_i * x_size + x_i];
    }
    else
    {
        M[tz][ty][tx] = 0.0f;
    }
    __syncthreads();

    if (tz < TILE_SIZE && ty < TILE_SIZE && tx < TILE_SIZE)
    {
        float value = 0.0f;
        for (int i = 0; i < MASK_SIZE; ++i)
        {
            for (int j = 0; j < MASK_SIZE; ++j)
            {
                for (int k = 0; k < MASK_SIZE; ++k)
                {
                    value += M[tz + i][ty + j][tx + k] * K[i][j][k];
                }
            }
        }
        if (z_o < z_size && y_o < y_size && x_o < x_size)
        {
            output[z_o * y_size * x_size + y_o * x_size + x_o] = value;
        }
    }
}

int main(int argc, char *argv[])
{
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength;
    float *hostInput;
    float *hostKernel;
    float *hostOutput;
    float *deviceInput;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel =
        (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];
    wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == 27);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first three elements were the dimensions
    wbCheck(cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and
    // do not need to be copied to the gpu
    wbCheck(cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpyToSymbol(K, hostKernel, kernelLength * sizeof(float)));
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ Initialize grid and block dimensions here
    dim3 DimGrid((x_size - 1) / TILE_SIZE + 1, (y_size - 1) / TILE_SIZE + 1, (z_size - 1) / TILE_SIZE + 1);
    dim3 DimBlock(INPUT_TILE_SIZE, INPUT_TILE_SIZE, INPUT_TILE_SIZE);

    //@@ Launch the GPU kernel here
    conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    wbCheck(cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}
