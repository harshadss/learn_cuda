#include<iostream>

void add(int n, float* x, float* y, float* z) {
    for (auto i = 0 ; i < n ; i++) {
        z[i] = x[i] + y[i];
    }
}

int main(void) {

    const unsigned int N = 10;

    float x[N], y[N];

    float* z = new float[N];

    for (auto i = 0 ; i < N; i++) {
        x[i] = i;
        y[i] = i;
    }

    add(N, x, y, z);

    for (auto i = 0; i < N ; i++) {
        std::cout << "i: " << i << "\tz: " << z[i] << std::endl;
    }
}