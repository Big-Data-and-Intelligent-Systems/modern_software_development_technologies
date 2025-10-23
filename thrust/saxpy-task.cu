#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

void usage(const char* filename)
{
    printf("Calculating a saxpy transform for two random vectors of the given length.\n");
    printf("Usage: %s <n>\n", filename);
}

using namespace thrust;

struct saxpy
{
    float a;
    // Constructor:
    saxpy(float a): a(a) {}
    
    // Operator for SAXPY: z = a * x + y
    __host__ __device__
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

int main(int argc, char* argv[])
{
    const int printable_n = 128;

    if (argc != 2)
    {
        usage(argv[0]);
        return 0;
    }

    int n = atoi(argv[1]);
    if (n <= 0)
    {
        usage(argv[0]);
        return 0;
    }
    
    cudaSetDevice(2);

    // Generate 3 vectors on host (z = a * x + y)
    thrust::host_vector<float> X(n), Y(n), Z(n);
    
    // Initialize with random values
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for(int i = 0; i < n; i++) {
        X[i] = dist(rng) * RAND_MAX;
        Y[i] = dist(rng) * RAND_MAX;
    }

    // Print out the input data if n is small.
    if (n <= printable_n)
    {
        printf("Input data:\n");
        for (int i = 0; i < n; i++)
            printf("%f   %f\n", 1.f*X[i] / RAND_MAX, 1.f*Y[i] / RAND_MAX);
        printf("\n");
    }

    // Transfer data to the device.
    thrust::device_vector<float> d_X = X;
    thrust::device_vector<float> d_Y = Y;
    thrust::device_vector<float> d_Z(n);

    float a = 2.5f;
    
    // Use transform to make an saxpy operation
    thrust::transform(d_X.begin(), d_X.end(), d_Y.begin(), d_Z.begin(), saxpy(a));

    // Transfer data back to host.
    Z = d_Z;

    // Print out the output data if n is small.
    if (n <= printable_n)
    {
        printf("Output data:\n");
        for (int i = 0; i < n; i++)
            printf("%f * %f + %f = %f\n", a, 1.f*X[i] / RAND_MAX, 1.f*Y[i] / RAND_MAX, Z[i] / RAND_MAX);
        printf("\n");
    }
    
    // Verify results for small n
    if (n <= printable_n) {
        printf("Verification:\n");
        for (int i = 0; i < n; i++) {
            float expected = a * X[i] + Y[i];
            float actual = Z[i];
            printf("Element %d: expected = %f, actual = %f, diff = %f\n", 
                   i, expected, actual, std::abs(expected - actual));
        }
    }

    return 0;
}