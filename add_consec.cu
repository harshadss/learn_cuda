#include<iostream>


__global__
void addConsec(float *x, float *y, unsigned int N) {

    unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x * 2;
    if (i < N) {
        y[i] = x[i] + x[i+1];
    }
}


int main(void) {

    constexpr unsigned int N = 1024;

    float *x, *y;

    // use unified memory
    
    cudaMallocManaged((void**)&x, N*sizeof(float));
    cudaMallocManaged((void**)&y, (N/2)*sizeof(float));

    // add dummy data
    // even data points get 2, odd data points get 1
    for (auto i = 0; i < N ; i++) {
        if ((i % 2) == 0) {
            x[i] = 2.0f;
        } else {
            x[i] = 1.0f;
        }
    }

    unsigned int threadsinBlock = 64;

    dim3 blockConfig = dim3(threadsinBlock);
    unsigned int numBlocks = N/128; // note 64 * 2

    dim3 gridConfig = dim3(numBlocks);

    addConsec<<<gridConfig, blockConfig>>>(x, y, N);

    cudaDeviceSynchronize();

    // check the numbers are correct
    float totalError = 0.0f;

    for (auto i = 0 ; i < N/2 ; i++) {
        if (y[i] != 3.0) {
            totalError += 1.0f;
        }
        std::cout << "i: " << i << "y: " << y[i] << std::endl;
    }

    cudaFree(x);
    cudaFree(y);

    std::cout << "Total Error: " << totalError << std::endl;
}