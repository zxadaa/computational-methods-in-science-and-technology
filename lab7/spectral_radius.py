import csv
import numpy as np

def generate_matrix_A(n, k=7, m=0.5):
    """Generuje macierz A zadaną w zadaniu."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = k
            else:
                i_math = i + 1
                j_math = j + 1
                A[i, j] = m / (n - i_math - j_math + 0.5)
    return A


def get_spectral_radius(A):
    """
    Oblicza promień spektralny macierzy iteracji dla metody Jacobiego.
    M_jacobi = -D^(-1) * R
    """
    n = A.shape[0]

    D_diag = np.diag(A)

    D_inv_diag = 1.0 / D_diag

    M = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = -D_inv_diag[i] * A[i, j]

    #wartości własne
    eigenvalues = np.linalg.eigvals(M)

    spectral_radius = np.max(np.abs(eigenvalues))

    return spectral_radius


if __name__ == '__main__':
    n_sizes = [2, 3, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 5000]
    filename = "results_task2.csv"

    with open(filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")

        writer.writerow([
            "N",
            "Spectral_Radius",
            "Is_Convergent"
        ])

        for n in n_sizes:
            A = generate_matrix_A(n)
            rho_s = get_spectral_radius(A)

            is_convergent = "YES" if rho_s < 1.0 else "NO"

            writer.writerow([
                n,
                round(rho_s, 6),
                is_convergent
            ])

    print(f"wniki zapisano do pliku: {filename}")