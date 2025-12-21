#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <curand_kernel.h>

__device__ double f(double x) {
    return (x*x)/(x + 1.0) + 1.0/x;
}

__global__ void monte_carlo_kernel(double a, double b, double max_f, int n, unsigned int seed, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        double x = a + (b - a) * curand_uniform_double(&state);
        double y = max_f * curand_uniform_double(&state);
        double fx = f(x);
        
        results[idx] = (y <= fx && y >= 0) ? 1 : 0;
    }
}

struct sum_func {
    __host__ __device__ int operator()(int a, int b) const {
        return a + b;
    }
};

double monte_carlo_cuda(double a, double b, int n, int threads_per_block = 256) {
    double max_f = 0;
    int steps = 1000;
    double step = (b - a) / steps;
    for (int i = 0; i <= steps; i++) {
        double x = a + i * step;
        double fx = (x*x)/(x + 1.0) + 1.0/x;
        if (fx > max_f) max_f = fx;
    }
    
    max_f *= 1.1;
    
    thrust::device_vector<int> d_results(n);
    int* d_results_ptr = thrust::raw_pointer_cast(d_results.data());
    
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    monte_carlo_kernel<<<blocks, threads_per_block>>>(a, b, max_f, n, time(0), d_results_ptr);
    cudaDeviceSynchronize();
    
    int total_hits = thrust::reduce(d_results.begin(), d_results.end(), 0, sum_func());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double area = (b - a) * max_f * total_hits / n;
    
    std::cout << "CUDA время: " << duration.count() << " мс" << std::endl;
    return area;
}

int main() {
    double a = 1.0, b = 4.0;
    std::cout << "Вычисление площади криволинейной трапеции" << std::endl;
    std::cout << "f(x) = x^2/(x+1) + 1/x на отрезке [" << a << ", " << b << "]" << std::endl;
    std::cout << std::endl;
    
    int test_cases[] = {100, 1000, 10000, 100000};
    
    for (int n : test_cases) {
        std::cout << "n = " << n << ":" << std::endl;
        double area = monte_carlo_cuda(a, b, n);
        std::cout << "Приближенная площадь: " << std::fixed << std::setprecision(6) << area << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}