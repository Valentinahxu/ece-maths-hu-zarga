import numpy as np

def triangleSup(T, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = 0
        for k in range(i+1, n):
            s = s + T[i, k] * x[k]
        x[i] = (b[i] - s )/ T[i, i]
    return x

def triangleInf(T, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        s = 0
        for k in range(i):
            s = s + T[i, k] * x[k]
        x[i] = (b[i] - s )/ T[i, i]
    return x

def decompLU(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    L[1, 1] = 1
    for j in range(n):
        U[1, j] = A[1, j]
    for i in range(2, n):
        L[i, 1] = A[i, 1]/U[1, 1]
    for i in range(2, n):
        L[i, i] = 1
        U[i, i] = A[i, i] - L[i, 1: i-1] * U[1: i-1, i]
        for j in range(i+1, n):
            U[i, j] = A[i, j] - L[i, 1: i-1] * U[1: i-1, j]
            L[j, i] = (A[j, i] - L[j, 1: i-1] * U[1: i-1, i])/U[i, i]
    return (L, U)

if __name__ == '__main__':
    T = np.array([[1, 0, 0], [4, 2, 0], [60, 5, 3]])
    b = np.array([1, 12, 1])
    x = triangleInf(T, b)
    print('x: ', x)
    A = np.array([[3, 2, 1], [6, 6, 3], [-3, 6, 4]])
    (L, U) = decompLU(A)
    print('L: ', L)
    print('U: ', U)
    (n, m) = np.shape(T)