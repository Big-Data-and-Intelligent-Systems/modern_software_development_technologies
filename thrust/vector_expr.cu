#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <iostream>
#include <cstdlib>

struct linear_combination
{
    float a, b, c;

    linear_combination(float a, float b, float c) : a(a), b(b), c(c) {}

    // Оператор для вычисления a*x + b*y + c*z
    __host__ __device__
    float operator()(const thrust::tuple<float, float, float>& t) const
    {
        float x = thrust::get<0>(t);
        float y = thrust::get<1>(t);
        float z = thrust::get<2>(t);
        return a * x + b * y + c * z;
    }
};

int main(int argc, char* argv[])
{
    int n;
    float a, b, c;

    std::cout << "Введите размер векторов n: ";
    std::cin >> n;
    std::cout << "Введите коэффициенты a, b, c: ";
    std::cin >> a >> b >> c;

    // Инициализация случайных чисел
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 10.0f);

    // Создание векторов на хосте
    thrust::host_vector<float> X(n), Y(n), Z(n);

    for (int i = 0; i < n; i++) {
        X[i] = dist(rng);
        Y[i] = dist(rng);
        Z[i] = dist(rng);
    }

    // Копируем на устройство
    thrust::device_vector<float> d_X = X;
    thrust::device_vector<float> d_Y = Y;
    thrust::device_vector<float> d_Z = Z;
    thrust::device_vector<float> d_D(n);

    // Применяем transform с итератором
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_X.begin(), d_Y.begin(), d_Z.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_X.end(), d_Y.end(), d_Z.end())),
        d_D.begin(),
        linear_combination(a, b, c)
    );

    thrust::host_vector<float> D = d_D;

    std::cout << "\nРезультаты (первые 10 элементов):\n";
    for (int i = 0; i < std::min(n, 10); i++) {
        std::cout << "D[" << i << "] = " << a << "*" << X[i]
                  << " + " << b << "*" << Y[i]
                  << " + " << c << "*" << Z[i]
                  << " = " << D[i] << "\n";
    }

    return 0;
}
