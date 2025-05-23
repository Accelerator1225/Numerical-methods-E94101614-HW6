
import numpy as np

# -------------------------
# Q1. Gaussian Elimination with Pivoting
# -------------------------

def question1():
    A = np.array([
        [1.19, 2.11, 100.0, 1.12],
        [14.2, -0.112, 12.2, -3.44],
        [0.0, 100.0, 99.9, 2.15],
        [15.3, -0.11, 13.1, -4.16]
    ], dtype=float)

    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    n = len(b)
    Ab = np.hstack((A, b.reshape(-1, 1)))

    # Partial pivoting
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        for j in range(i + 1, n):
            factor = Ab[j][i] / Ab[i][i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    print("Q1: Gaussian Elimination with Pivoting Result")
    for i, val in enumerate(x):
        print(f"x{i+1} = {val:.6f}")
    print()

# -------------------------
# Q2. Matrix Inversion
# -------------------------

def question2():
    A = np.array([
        [4, 1, 1, 0],
        [1, 3, 1, 0],
        [1, 1, 6, 2],
        [0, 0, 2, 5]
    ], dtype=float)

    A_inv = np.linalg.inv(A)

    print("Q2: Inverse of Matrix A")
    for row in A_inv:
        print("  ".join(f"{val: .6f}" for val in row))
    print()

# -------------------------
# Q3. Crout Factorization for Tridiagonal Matrix
# -------------------------

def question3():
    n = 4
    a = np.array([0, 1, 1, 1], dtype=float)  # sub-diagonal (a[0] unused)
    b = np.array([3, 3, 3, 3], dtype=float)  # diagonal
    c = np.array([1, 1, 1, 0], dtype=float)  # super-diagonal (c[-1] unused)
    d = np.array([2, 3, 4, 1], dtype=float)  # RHS

    l = np.zeros(n)
    u = np.zeros(n - 1)
    y = np.zeros(n)
    x = np.zeros(n)

    # Crout decomposition
    l[0] = b[0]
    u[0] = c[0] / l[0]
    for i in range(1, n - 1):
        l[i] = b[i] - a[i] * u[i - 1]
        u[i] = c[i] / l[i]
    l[-1] = b[-1] - a[-1] * u[-2]

    # Forward substitution
    y[0] = d[0] / l[0]
    for i in range(1, n):
        y[i] = (d[i] - a[i] * y[i - 1]) / l[i]

    # Backward substitution
    x[-1] = y[-1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] - u[i] * x[i + 1]

    print("Q3: Crout Factorization Result")
    for i, val in enumerate(x):
        print(f"x{i+1} = {val:.6f}")
    print()

# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    question1()
    question2()
    question3()
