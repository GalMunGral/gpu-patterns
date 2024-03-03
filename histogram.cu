// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 1024
#define HISTO_GRID_SIZE 5

#define wbCheck(stmt)                                                      \
    do                                                                     \
    {                                                                      \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

//@@ insert code here
__global__ void float2uchar(float *floatImage, unsigned char *ucharImage, int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        ucharImage[i] = 255 * floatImage[i];
    }
}

__global__ void rgb2gray(unsigned char *ucharImage, unsigned char *grayImage, int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        unsigned char r = ucharImage[3 * i];
        unsigned char g = ucharImage[3 * i + 1];
        unsigned char b = ucharImage[3 * i + 2];
        grayImage[i] = 0.21 * r + 0.71 * g + 0.07 * b;
    }
}

__global__ void histo_kernel(unsigned char *grayImage, int *histo, int size)
{
    __shared__ int privateHisto[HISTOGRAM_LENGTH];
    if (threadIdx.x < HISTOGRAM_LENGTH)
    {
        privateHisto[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (i < size)
    {
        atomicAdd(&privateHisto[grayImage[i]], 1);
        i += stride;
    }
    __syncthreads();

    if (threadIdx.x < HISTOGRAM_LENGTH)
    {
        atomicAdd(&histo[threadIdx.x], privateHisto[threadIdx.x]);
    }
}

__global__ void cdf_kernel(int *histo, float *cdf, int size)
{
    __shared__ float cdf_s[HISTOGRAM_LENGTH];

    const int i = threadIdx.x;
    const int BlockSize = HISTOGRAM_LENGTH / 2;

    cdf_s[i] = histo[i] / (1.0 * size);
    cdf_s[i + BlockSize] = histo[i + BlockSize] / (1.0 * size);

    for (int stride = 1; stride <= BlockSize; stride *= 2)
    {
        __syncthreads();
        int index = (i + 1) * 2 * stride - 1;
        if (index < HISTOGRAM_LENGTH)
        {
            cdf_s[index] += cdf_s[index - stride];
        }
    }

    for (int stride = BlockSize / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        int index = (i + 1) * 2 * stride - 1;
        if (index + stride < 2 * BlockSize)
        {
            cdf_s[index + stride] += cdf_s[index];
        }
    }

    __syncthreads();
    cdf[i] = cdf_s[i];
    cdf[i + BlockSize] = cdf_s[i + BlockSize];
}

__device__ float clamp(float x, float start, float end)
{
    return min(max(x, start), end);
}

__device__ unsigned char correct_color(unsigned char val, float *cdf)
{
    const float cdfmin = cdf[0];
    return clamp(255 * (cdf[val] - cdfmin) / (1.0 - cdfmin), 0.0f, 255.0f);
}

__global__ void equalize(unsigned char *ucharImage, float *floatImage, float *cdf, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        floatImage[i] = correct_color(ucharImage[i], cdf) / 255.0;
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    //@@ Insert more code here
    float *floatImage;
    unsigned char *ucharImage;
    unsigned char *grayImage;
    int *histo;
    float *cdf;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    const int grayImageSize = imageWidth * imageHeight;
    const int rgbImageSize = imageWidth * imageHeight * imageChannels;

    wbCheck(cudaMalloc((void **)&floatImage, rgbImageSize * sizeof(float)));
    wbCheck(cudaMalloc((void **)&ucharImage, rgbImageSize));
    wbCheck(cudaMalloc((void **)&grayImage, grayImageSize));
    wbCheck(cudaMalloc((void **)&histo, HISTOGRAM_LENGTH * sizeof(int)));
    wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float)));

    wbCheck(cudaMemcpy(floatImage, hostInputImageData, rgbImageSize * sizeof(float), cudaMemcpyHostToDevice));

    float2uchar<<<(rgbImageSize - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(floatImage, ucharImage, rgbImageSize);
    rgb2gray<<<(grayImageSize - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(ucharImage, grayImage, grayImageSize);
    histo_kernel<<<HISTO_GRID_SIZE, BLOCK_SIZE>>>(grayImage, histo, grayImageSize);
    cdf_kernel<<<1, HISTOGRAM_LENGTH / 2>>>(histo, cdf, grayImageSize);
    equalize<<<(rgbImageSize - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(ucharImage, floatImage, cdf, rgbImageSize);

    wbCheck(cudaMemcpy(hostOutputImageData, floatImage, rgbImageSize * sizeof(float), cudaMemcpyDeviceToHost));

    wbImage_setData(outputImage, hostOutputImageData);

    wbSolution(args, outputImage);

    //@@ insert code here
    wbCheck(cudaFree(floatImage));
    wbCheck(cudaFree(ucharImage));
    wbCheck(cudaFree(grayImage));
    wbCheck(cudaFree(histo));
    wbCheck(cudaFree(cdf));
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
