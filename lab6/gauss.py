import csv
import time
import numpy as np

def generate_true_x(n):
    """
    Generates a true solution vector with random 1 and -1.
    """
    np.random.seed(47654678)
    return np.random.choice([1.0, -1.0], size=n)


def calculate_max_error(x_calculated, x_true):
    """Calculates the maximum norm error between the calculated and true vectors."""
    return np.max(np.abs(x_calculated - x_true))


def gaussian_elimination(A, b):
    """
    Solves the system of linear equations Ax = b using Gaussian elimination
    with partial pivoting.
    """
    A = np.array(A, dtype=A.dtype)
    b = np.array(b, dtype=b.dtype)
    n = len(b)

    #1.forward elimination
    for i in range(n - 1):
        p = i + np.argmax(np.abs(A[i:n, i]))

        if A[p, i] == 0:
            raise ValueError("brak unikalnego rozwiązania - macierz osobliwa")

        if p != i:
            A[[i, p]] = A[[p, i]]
            b[[i, p]] = b[[p, i]]

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]
            b[j] = b[j] - m * b[i]

    if A[n - 1, n - 1] == 0:
        raise ValueError("brak unikalnego rozwiązania - macierz osobliwa")

    #2.backward substitution
    x = np.zeros(n, dtype=b.dtype)
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = np.sum(A[i, i + 1:] * x[i + 1:])
        x[i] = (b[i] - suma) / A[i, i]

    return x


def create_task1_matrix(n, dtype=np.float64):
    """Creates the ill-conditioned matrix for Task 1."""
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            if i == 0:
                A[i, j] = 1.0
            else:
                A[i, j] = 1.0 / (i + j + 1.0)
    return A


def create_task2_matrix(n, dtype=np.float64):
    """Creates the well-conditioned matrix for Task 2."""
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            I = i + 1
            J = j + 1
            if J >= I:
                A[i, j] = (2.0 * I) / J
            else:
                A[i, j] = A[j, i]
    return A

def create_task3_vectors_a(n, k=7, m=3, dtype=np.float64):
    """Creates the tridiagonal matrix vectors for Task 3."""
    D = np.zeros(n, dtype=dtype)
    U = np.zeros(n - 1, dtype=dtype)
    L = np.zeros(n - 1, dtype=dtype)

    for i in range(n):
        I = i + 1
        D[i] = k
        if i < n - 1:
            U[i] = 1.0 / (I + m)
        if i > 0:
            L[i - 1] = k / (I + m + 1.0)

    return L, D, U


def thomas_algorithm(L, D, U, b):
    """
    Solves the system of equations for a tridiagonal matrix using the Thomas algorithm.
    """
    n = len(b)
    u_prime = np.zeros(n - 1, dtype=b.dtype)
    d_prime = np.zeros(n, dtype=b.dtype)
    x = np.zeros(n, dtype=b.dtype)

    #1.forward elimination
    u_prime[0] = U[0] / D[0]
    d_prime[0] = b[0] / D[0]

    for i in range(1, n - 1):
        denominator = D[i] - L[i - 1] * u_prime[i - 1]
        u_prime[i] = U[i] / denominator
        d_prime[i] = (b[i] - L[i - 1] * d_prime[i - 1]) / denominator

    denominator_last = D[n - 1] - L[n - 2] * u_prime[n - 2]
    d_prime[n - 1] = (b[n - 1] - L[n - 2] * d_prime[n - 2]) / denominator_last

    #2.backward substitution
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - u_prime[i] * x[i + 1]

    return x


def save_results_to_csv(filename, results, columns):
    """Saves a list of dictionaries to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, delimiter=';')
        writer.writeheader()
        writer.writerows(results)
    print(f"results saved to {filename}")


def run_task1(filename="task1_results.csv"):
    sizes = list(range(2, 21))
    results = []

    for n in sizes:
        x_true = generate_true_x(n)

        A_32 = create_task1_matrix(n, dtype=np.float32)
        x_true_32 = x_true.astype(np.float32)
        b_32 = A_32 @ x_true_32
        x_calc_32 = gaussian_elimination(A_32, b_32)

        A_64 = create_task1_matrix(n, dtype=np.float64)
        b_64 = A_64 @ x_true
        x_calc_64 = gaussian_elimination(A_64, b_64)

        results.append({
            "n": n,
            "cond_A_float32": np.linalg.cond(A_32),
            "cond_A_float64": np.linalg.cond(A_64),
            "error_float32": calculate_max_error(x_calc_32, x_true_32),
            "error_float64": calculate_max_error(x_calc_64, x_true)
        })

    columns = ["n", "cond_A_float32", "cond_A_float64", "error_float32", "error_float64"]
    save_results_to_csv(filename, results, columns)


def run_task2(filename="task2_results.csv"):
    sizes = list(range(2, 21)) + [50, 100, 150]
    results = []

    for n in sizes:
        x_true = generate_true_x(n)

        A_32 = create_task2_matrix(n, dtype=np.float32)
        x_true_32 = x_true.astype(np.float32)
        b_32 = A_32 @ x_true_32
        x_calc_32 = gaussian_elimination(A_32, b_32)

        A_64 = create_task2_matrix(n, dtype=np.float64)
        b_64 = A_64 @ x_true
        x_calc_64 = gaussian_elimination(A_64, b_64)

        results.append({
            "n": n,
            "cond_A_float32": np.linalg.cond(A_32),
            "cond_A_float64": np.linalg.cond(A_64),
            "error_float32": calculate_max_error(x_calc_32, x_true_32),
            "error_float64": calculate_max_error(x_calc_64, x_true)
        })

    columns = ["n", "cond_A_float32", "cond_A_float64", "error_float32", "error_float64"]
    save_results_to_csv(filename, results, columns)


def run_task3(filename="task3_results.csv"):
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 250, 500, 1000]
    results = []

    for n in sizes:
        x_true = generate_true_x(n)

        x_true_32 = x_true.astype(np.float32)
        L_32, D_32, U_32 = create_task3_vectors_a(n, dtype=np.float32)
        A_full_32 = np.diag(D_32) + np.diag(U_32, k=1) + np.diag(L_32, k=-1)
        b_32 = A_full_32 @ x_true_32

        cond_32 = np.linalg.cond(A_full_32)
        mem_gauss_32 = A_full_32.nbytes + b_32.nbytes
        mem_thomas_32 = L_32.nbytes + D_32.nbytes + U_32.nbytes + b_32.nbytes

        start_gauss = time.perf_counter()
        x_gauss_32 = gaussian_elimination(A_full_32, b_32)
        time_gauss_32 = time.perf_counter() - start_gauss

        start_thomas = time.perf_counter()
        x_thomas_32 = thomas_algorithm(L_32, D_32, U_32, b_32)
        time_thomas_32 = time.perf_counter() - start_thomas

        L_64, D_64, U_64 = create_task3_vectors_a(n, dtype=np.float64)
        A_full_64 = np.diag(D_64) + np.diag(U_64, k=1) + np.diag(L_64, k=-1)
        b_64 = A_full_64 @ x_true

        cond_64 = np.linalg.cond(A_full_64)
        mem_gauss_64 = A_full_64.nbytes + b_64.nbytes
        mem_thomas_64 = L_64.nbytes + D_64.nbytes + U_64.nbytes + b_64.nbytes

        start_gauss = time.perf_counter()
        x_gauss_64 = gaussian_elimination(A_full_64, b_64)
        time_gauss_64 = time.perf_counter() - start_gauss

        start_thomas = time.perf_counter()
        x_thomas_64 = thomas_algorithm(L_64, D_64, U_64, b_64)
        time_thomas_64 = time.perf_counter() - start_thomas

        results.append({
            "n": n,
            "cond_A_float32": cond_32,
            "cond_A_float64": cond_64,
            "error_gauss_float32": calculate_max_error(x_gauss_32, x_true_32),
            "error_gauss_float64": calculate_max_error(x_gauss_64, x_true),
            "error_thomas_float32": calculate_max_error(x_thomas_32, x_true_32),
            "error_thomas_float64": calculate_max_error(x_thomas_64, x_true),
            "time_gauss_float32_s": time_gauss_32,
            "time_gauss_float64_s": time_gauss_64,
            "time_thomas_float32_s": time_thomas_32,
            "time_thomas_float64_s": time_thomas_64,
            "mem_gauss_float32_bytes": mem_gauss_32,
            "mem_gauss_float64_bytes": mem_gauss_64,
            "mem_thomas_float32_bytes": mem_thomas_32,
            "mem_thomas_float64_bytes": mem_thomas_64
        })

    columns = [
        "n", "cond_A_float32", "cond_A_float64",
        "error_gauss_float32", "error_gauss_float64",
        "error_thomas_float32", "error_thomas_float64",
        "time_gauss_float32_s", "time_gauss_float64_s",
        "time_thomas_float32_s", "time_thomas_float64_s",
        "mem_gauss_float32_bytes", "mem_gauss_float64_bytes",
        "mem_thomas_float32_bytes", "mem_thomas_float64_bytes"
    ]
    save_results_to_csv(filename, results, columns)

if __name__ == "__main__":
    run_task3()