/* a simple 2D cuda kernel to transpose a matrix

 its kind of pointless from calculation standpoint because matrices are 1D memory
 in row major format here. So in real libs like pytorch, transpose does NOT physically
 change data in memory. It will be implemented in terms of dimensions and strides of
 the array.

the idea of this toy code is to understand 2D grids in Cuda */


#include <iostream>
#include <iomanip>

__global__
void transpose(float *in, float *out, unsigned int rows, unsigned int cols) {

    unsigned int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    unsigned int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    unsigned int out_row = col;
    unsigned int out_col = row;

    if ((row < rows) && (col < cols)) {
        unsigned int in_offset = (row * cols) + col;
        unsigned int out_offset = (out_row * rows) + out_col;
        out[out_offset] = in[in_offset];
    }
}

int main(void)
{
    // create a simple matrix of size 768 * 1024
    const unsigned int rows = 768;
    const unsigned int cols = 1024;
    constexpr int total_size = rows * cols;
    float *in = new float[total_size];
    float *out = new float[total_size];

    // fill it up with data such that we can track transpose is correct
    for (auto r = 0; r < rows; r++)
    {
        for (auto c = 0; c < cols; c++)
        {
            auto offset = (r * cols) + c;
            in[offset] = (float)c;
        }
    }

    // print it for first few rows
    
    for (auto r = 0; r < 8; r++)
    {
        
        for (auto c = 0; c < 16; c++)
        {
            auto offset = (r * cols) + c;
            std::cout << std::setfill('0') << std::setw(4) << in[offset] << " " ;
        }
        std::cout << std::endl << std::endl;
    }

    dim3 blockDim {16, 16, 1}; // 16 * 16 threads in each block
    
    unsigned int col_blocks = cols/16;
    unsigned int row_blocks = rows/16;

    dim3 gridDim {col_blocks, row_blocks, 1}; // note : x and then y

    // moving host mem to cuda. @TODO explore cudaManaged for this
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, total_size * sizeof(float));
    cudaMalloc((void **)&d_out, total_size * sizeof(float));

    cudaMemcpy(d_in, in, total_size*sizeof(float), cudaMemcpyHostToDevice);

    transpose<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);

    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(out, d_out, total_size*sizeof(float), cudaMemcpyDeviceToHost);

    // print result
    std::cout << "Printing result: " << std::endl;
    
    for (auto out_r = 0; out_r < 16; out_r++)
    {
        
        for (auto out_c = 0; out_c < 8 ; out_c++)
        {
            auto offset = (out_r * rows) + out_c;
            std::cout << std::setfill('0') << std::setw(4) << out[offset] << " " ;
        }
        std::cout << std::endl << std::endl;
    }


    // cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] in;
    delete[] out;
}