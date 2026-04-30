import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
import time

# ПАРАМЕТРЫ
alpha = 1e-5
Lx = Ly = 1.0
Nx = Ny = 100

hx = Lx / Nx
hy = Ly / Ny

tau = 2.0
timefin = 200
tol = 1e-6

steps = int(timefin / tau)

# НАЧАЛЬНОЕ УСЛОВИЕ
def init():
    T = np.ones((Nx+1, Ny+1)) * 100.0

    # Граничные условия (Дирихле)
    T[0, :] = 1.0
    T[-1, :] = 10.0
    T[:, 0] = 10.0
    T[:, -1] = 10.0

    return T

# КОЭФФИЦИЕНТЫ
ae = alpha * tau / hx**2
aw = ae
an = alpha * tau / hy**2
as_ = an
ap = ae + aw + an + as_

# MATVEC (синхронный)
def matvec(T):
    res = np.zeros_like(T)
    for i in range(1, Nx):
        for j in range(1, Ny):
            res[i, j] = (1 + ap) * T[i, j] \
                        - ae * T[i+1, j] - aw * T[i-1, j] \
                        - an * T[i, j+1] - as_ * T[i, j-1]
    return res

# MATVEC (параллельный)
@njit(parallel=True)
def matvec_parallel(T):
    res = np.zeros_like(T)
    for i in prange(1, Nx):
        for j in range(1, Ny):
            res[i, j] = (1 + ap) * T[i, j] \
                        - ae * T[i+1, j] - aw * T[i-1, j] \
                        - an * T[i, j+1] - as_ * T[i, j-1]
    return res

# =========================
# CG (синхронный)
# =========================
def cg_solver(b, x0):
    x = x0.copy()
    r = b - matvec(x)
    p = r.copy()

    rsold = np.sum(r * r)

    for k in range(1000):
        Ap = matvec(p)

        alpha_cg = rsold / np.sum(p * Ap)
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap

        rsnew = np.sum(r * r)

        if np.sqrt(rsnew) < tol:
            return x, k

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, 1000

# =========================
# CG (параллельный)
# =========================
@njit(parallel=True)
def cg_solver_parallel(b, x0):
    x = x0.copy()
    r = b - matvec_parallel(x)
    p = r.copy()

    rsold = 0.0
    for i in prange(1, Nx):
        for j in range(1, Ny):
            rsold += r[i, j] * r[i, j]

    for k in range(1000):
        Ap = matvec_parallel(p)

        pAp = 0.0
        for i in prange(1, Nx):
            for j in range(1, Ny):
                pAp += p[i, j] * Ap[i, j]

        alpha_cg = rsold / pAp

        for i in prange(1, Nx):
            for j in range(1, Ny):
                x[i, j] += alpha_cg * p[i, j]
                r[i, j] -= alpha_cg * Ap[i, j]

        rsnew = 0.0
        for i in prange(1, Nx):
            for j in range(1, Ny):
                rsnew += r[i, j] * r[i, j]

        if np.sqrt(rsnew) < tol:
            return x, k

        beta = rsnew / rsold

        for i in prange(1, Nx):
            for j in range(1, Ny):
                p[i, j] = r[i, j] + beta * p[i, j]

        rsold = rsnew

    return x, 1000

# =========================
# ВРЕМЕННОЙ ШАГ
# =========================
def time_step(T):
    b = T.copy()
    T_new, iters = cg_solver(b, T)

    T_new[0, :] = 1.0
    T_new[-1, :] = 10.0
    T_new[:, 0] = 10.0
    T_new[:, -1] = 10.0

    return T_new, iters

def time_step_parallel(T):
    b = T.copy()
    T_new, iters = cg_solver_parallel(b, T)

    T_new[0, :] = 1.0
    T_new[-1, :] = 10.0
    T_new[:, 0] = 10.0
    T_new[:, -1] = 10.0

    return T_new, iters

# СИНХРОННЫЙ ЗАПУСК
print("Синхронный расчет...")
T_sync = init()

start = time.time()
iters_sync = []

for _ in tqdm(range(steps), desc="Синхронный"):
    T_sync, it = time_step(T_sync)
    iters_sync.append(it)

time_sync = time.time() - start

# ПАРАЛЛЕЛЬНЫЙ ЗАПУСК
print("Параллельный расчет...")
T_par = init()

start = time.time()
iters_par = []

for _ in tqdm(range(steps), desc="Параллельный"):
    T_par, it = time_step_parallel(T_par)
    iters_par.append(it)

time_par = time.time() - start

print(f"\nСинхронный: {time_sync:.2f} сек")
print(f"Параллельный: {time_par:.2f} сек")
print(f"Ускорение: {time_sync / time_par:.2f}x")
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(T_sync.T, 20, cmap='hot')
plt.title("Синхронный")
plt.subplot(1,2,2)
plt.contourf(T_par.T, 20, cmap='hot')
plt.title("Параллельный")
plt.show()