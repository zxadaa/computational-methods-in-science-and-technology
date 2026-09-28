import csv
import numpy as np
import time

def manual_mat_vec_mul(A, x):
    """
    Mnożenie macierzy przez wektor (O(N^2)).
    """
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(n):
            sum_val += A[i, j] * x[j]
        result[i] = sum_val
    return result

def generate_matrices(n, k=7, m=0.5):
    """
    Generuje macierz A, wektor referencyjny x_ref oraz wektor wyrazów wolnych b
    zgodnie z wariantem.
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = k
            else:
                i_math = i + 1
                j_math = j + 1
                A[i, j] = m / (n - i_math - j_math + 0.5)

    x_ref = np.random.choice([1, -1], size=n)

    b = manual_mat_vec_mul(A, x_ref)

    return A, x_ref, b

def jacobi(A, b, rho, x0, criterion=1, max_iter=10000):
    """
    Metoda Jacobiego.
    criterion 1: ||x_new - x|| < rho
    criterion 2: ||Ax - b|| < rho
    """
    n = len(b)
    x = x0.copy()

    start_time = time.time()
    iterations = 0

    while iterations < max_iter:
        x_new = np.zeros(n)

        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    sum_val += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_val) / A[i, i]

        if criterion == 1:
            error = np.linalg.norm(x_new - x, ord=2)
        elif criterion == 2:
            Ax = manual_mat_vec_mul(A, x_new)
            error = np.linalg.norm(Ax - b, ord=2)
        else:
            raise ValueError("nieznane kryterium")

        x = x_new.copy()
        iterations += 1

        if error < rho:
            break

    execution_time = time.time() - start_time
    time_per_iter = execution_time / iterations if iterations > 0 else 0

    return x, iterations, execution_time, time_per_iter


if __name__ == '__main__':
    np.random.seed(47654678)
    n_sizes = [2, 3, 5, 10, 15, 20, 50, 100, 200, 500]
    rho_values = [1e-3, 1e-5, 1e-9, 1e-15]

    filename = "results_task1.csv"

    with open(filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")

        writer.writerow([
            "N",
            "Rho",
            "Initial_Vector",
            "Criterion",
            "Iterations",
            "Total_Time_s",
            "Time_per_Iteration_s",
            "Real_Error"
        ])

        for n_size in n_sizes:
            for rho_test in rho_values:
                A, x_ref, b = generate_matrices(n_size)

                x0_close = np.zeros(n_size)
                x0_far = np.random.choice([-100.0, 100.0], size=n_size)

                initial_vectors = [
                    ("Close", x0_close),
                    ("Far", x0_far)
                ]

                for name, x0 in initial_vectors:
                    x1, iter1, time1, tpi1 = jacobi(A, b, rho_test, x0, criterion=1)
                    real_error1 = np.max(np.abs(x1 - x_ref))

                    writer.writerow([
                        n_size,
                        rho_test,
                        name,
                        "Incremental (1)",
                        iter1,
                        round(time1, 6),
                        round(tpi1, 8),
                        f"{real_error1:.2e}"
                    ])

                    x2, iter2, time2, tpi2 = jacobi(A, b, rho_test, x0, criterion=2)
                    real_error2 = np.max(np.abs(x2 - x_ref))

                    writer.writerow([
                        n_size,
                        rho_test,
                        name,
                        "Residual (2)",
                        iter2,
                        round(time2, 6),
                        round(tpi2, 8),
                        f"{real_error2:.2e}"
                    ])
            print(f"zakończono dla n = {n_size}")

    print(f"wyniki zapisano do pliku: {filename}")