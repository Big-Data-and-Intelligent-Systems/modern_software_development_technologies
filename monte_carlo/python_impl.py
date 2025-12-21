import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def f(x: float) -> float:
    """Функция f(x) = x^2/(x+1) + 1/x"""
    return (x**2)/(x + 1.0) + 1.0/x

def monte_carlo_sequential(a: float, b: float, n: int) -> Tuple[float, float]:
    x_samples = np.linspace(a, b, 1000)
    f_samples = f(x_samples)
    max_f = np.max(f_samples) * 1.1  
    
    # Измеряем время
    start_time = time.time()
    
    np.random.seed(42)  
    x_points = a + (b - a) * np.random.random(n)
    y_points = max_f * np.random.random(n)
    
    f_values = f(x_points)
    
    hits = np.sum((y_points <= f_values) & (y_points >= 0))
    
    area = (b - a) * max_f * hits / n
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  
    
    return area, execution_time

def main():
    a, b = 1.0, 4.0
    print("Вычисление площади криволинейной трапеции")
    print(f"f(x) = x^2/(x+1) + 1/x на отрезке [{a}, {b}]")
    print()
    
    test_cases = [100, 1000, 10000, 100000]
    results = []
    
    for n in test_cases:
        print(f"n = {n}:")
        area, exec_time = monte_carlo_sequential(a, b, n)
        print(f"  Приближенная площадь: {area:.6f}")
        print(f"  Время выполнения: {exec_time:.2f} мс")
        results.append((n, area, exec_time))
        print()
    
    return results

def plot_results(cuda_times: List[float], python_times: List[float], 
                  test_cases: List[int]):
    plt.figure(figsize=(10, 6))
    
    plt.plot(test_cases, python_times, 'b-o', label='Python (последовательный)', linewidth=2)
    plt.plot(test_cases, cuda_times, 'r-s', label='CUDA (параллельный)', linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Количество точек (n)', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.title('Сравнение производительности методов Монте-Карло', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    for i, n in enumerate(test_cases):
        plt.annotate(f'{python_times[i]:.1f} мс', 
                    xy=(n, python_times[i]), 
                    xytext=(5, 5), textcoords='offset points')
        plt.annotate(f'{cuda_times[i]:.1f} мс', 
                    xy=(n, cuda_times[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_performance.png', dpi=300)

if __name__ == "__main__":
    print("=" * 60)
    print("ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ НА PYTHON")
    print("=" * 60)
    python_results = main()
    
    test_cases = [100, 1000, 10000, 100000]
    
    cuda_times_example = [2.1, 2.3, 2.8, 4.5]  # значения с моей видюхи
    
    python_times = [r[2] for r in python_results]
    
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    print(f"{'n':>10} | {'Python (мс)':>12} | {'CUDA (мс)':>10} | {'Ускорение':>10}")
    print("-" * 60)
    
    for i, n in enumerate(test_cases):
        speedup = python_times[i] / cuda_times_example[i] if i < len(cuda_times_example) else 0
        print(f"{n:10} | {python_times[i]:12.2f} | {cuda_times_example[i]:10.2f} | {speedup:10.2f}x")

    plot_results(cuda_times_example, python_times, test_cases)