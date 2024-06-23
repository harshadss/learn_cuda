#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    constexpr int N {1 << 20};
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (auto i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    unsigned int blockSize {-256};
    unsigned int numBlocks = (N + blockSize - 1) / blockSize;

    dim3 gridConfig {numBlocks};
    dim3 blockConfig {blockSize};

    add<<<gridConfig, blockConfig>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (auto i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}