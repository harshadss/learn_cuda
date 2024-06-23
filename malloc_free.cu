#include<iostream>
#include<unistd.h>

int main(void) {

    constexpr unsigned int N = 1024;

    float *y ; // uninitialised pointer to float
    float *x = new float[N];

    for (auto i = 0; i < N ; i++) {
        x[i] = 1.0;
    }

    unsigned int size = N * sizeof(float);

    cudaMalloc((void**)&y, size);

    cudaMemcpy(y, x, size, cudaMemcpyHostToDevice);

    sleep(3);

    cudaFree(y);

    delete[] x;
}