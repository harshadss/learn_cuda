#include<iostream>
#include<unistd.h>

__global__ void add(float* h_x, float* h_y, float* h_z, unsigned int N) {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (index < N) {
        h_z[index] = h_x[index] + h_y[index];
    }
}

int main(void) {

    const unsigned int N = 1024;

    // create local data
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_z = new float[N];

    for (auto i = 0; i < N; i++) {
        h_x[i] = 1.0;
        h_y[i] = 2.0;
    }

    // create pointers to hold device memory
    float* d_x; 
    float* d_y; 
    float* d_z;

    sleep(3);
    
    // allocate space on device
    cudaMalloc((void**)&d_x, N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(float));
    cudaMalloc((void**)&d_z, N*sizeof(float));
    // move local data to device
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
    // launch kernel
    // kernel config
    unsigned int numThreadsInBlock = 32;
    unsigned int numBlocks = (N + numThreadsInBlock - 1)/numThreadsInBlock;

    dim3 blockConfig {numThreadsInBlock};
    dim3 gridConfig {numBlocks};

    add<<<gridConfig, blockConfig>>>(d_x, d_y, d_z, N);

    cudaDeviceSynchronize();
    // move back device to host
    cudaMemcpy(h_z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    sleep(3);
    // free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    sleep(3);
    // validate answer
    for (auto i = 0; i < 10; i++) {
        std::cout << "i: " << i << "\th_z[i]: " << h_z[i] << std::endl;
    }
    // free host memory
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;
    return 0;
}