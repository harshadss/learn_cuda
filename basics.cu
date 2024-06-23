#include<iostream>


int main(void) {

    constexpr unsigned int N = 3 * 2;

    uint3 data {1, 2, 3};

    dim3 data1 {N}; 

    std::cout << data.x << "\t" << data.y << "\t" << data.z << std::endl;

    std::cout << data1.x << "\t" << data1 .y << "\t" << data1 .z << std::endl;
}