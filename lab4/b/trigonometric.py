import numpy as np
import matplotlib.pyplot as plt
import os
import csv

#WZÓR FUNKCJI
def f(x, k=5, m=0.5):
    return np.sin(k * x / np.pi) * np.exp(-m * x / np.pi)

#GENEROWANIE WĘZŁÓW RÓWNOMIERNYCH
def generate_uniform_nodes(a, b, n, include_endpoint=True):
    return np.linspace(a, b, n, endpoint=include_endpoint)

#APROKSYMACJA ŚREDNIOKWADRATOWA - WIELOMIANY TRYGONOMETRYCZNE
def discrete_trigonometric_least_squares_fit(x, y, m, a, b, weights=None):
    """
    Funkcja obliczająca współczynniki aproksymacji trygonometrycznej.
    funkcje bazowe: 1, cos(cx), sin(cx), ..., cos(mcx), sin(mcx), liczba - 2m + 1
    gdzie c = 2pi / (b - a)
    """
    n = len(x)
    if weights is None:
        weights = np.ones(n)

    #współczynnik skalujący przedział [a, b]
    c = 2 * np.pi / (b - a)
    num_funcs = 2 * m + 1

    A = np.zeros((n, num_funcs))
    A[:, 0] = 1.0
    for k in range(1, m + 1):
        A[:, 2 * k - 1] = np.cos(k * c * (x - a))
        A[:, 2 * k] = np.sin(k * c * (x - a))

    W = np.diag(weights)
    G = A.T @ W @ A
    B = A.T @ W @ y

    coeffs = np.linalg.solve(G, B)

    def evaluate(x_val):
        x_val = np.asarray(x_val)
        y_val = np.ones_like(x_val, dtype=float) * coeffs[0]
        for k in range(1, m + 1):
            y_val += coeffs[2 * k - 1] * np.cos(k * c * (x_val - a))
            y_val += coeffs[2 * k] * np.sin(k * c * (x_val - a))
        return y_val

    return evaluate, coeffs

#FUNKCJE POMOCNICZE
def calculate_mse_error(y_true, y_interp):
    return np.mean((y_true - y_interp) ** 2)

def calculate_max_error(y_true, y_interp):
    return np.max(np.abs(y_true - y_interp))

def save_plot(fig, filename):
    fig.savefig(filename, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    INCLUDE_ENDPOINT = False

    folder_name = "wykresy" if INCLUDE_ENDPOINT else "wykresy_epf"
    csv_filename = "results_trig.csv" if INCLUDE_ENDPOINT else "results_trig_epf.csv"

    os.makedirs(folder_name, exist_ok=True)

    #parametry zadania
    a_val, b_val = -np.pi * np.pi, np.pi * np.pi

    n_list = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    m_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 20, 40]  #rzędy aproksymacji

    x_eval = np.linspace(a_val, b_val, 500)
    y_true = f(x_eval)

    results = []

    for n in n_list:
        x_nodes = generate_uniform_nodes(a_val, b_val, n, include_endpoint=INCLUDE_ENDPOINT)
        y_nodes = f(x_nodes)

        for m in m_list:
            row = {'n_nodes': n, 'm_degree': m}

            if 2 * m + 1 <= n:
                approx_func_tryg, _ = discrete_trigonometric_least_squares_fit(x_nodes, y_nodes, m, a_val, b_val)
                y_approx_tryg = approx_func_tryg(x_eval)

                row['tryg_max_err'] = calculate_max_error(y_true, y_approx_tryg)
                row['tryg_mse_err'] = calculate_mse_error(y_true, y_approx_tryg)

                fig_tryg, ax_tryg = plt.subplots(figsize=(8, 5))
                ax_tryg.plot(x_eval, y_true, 'k--', label='oryginał f(x)')
                ax_tryg.plot(x_eval, y_approx_tryg, 'r-', label='aprok. trygonometryczna')
                ax_tryg.plot(x_nodes, y_nodes, 'ko', markersize=4, label='węzły')

                ax_tryg.set_title(f'Aproksymacja trygonometryczna: n={n}, m={m}')
                ax_tryg.set_ylim(np.min(y_true) - 1, np.max(y_true) + 1)
                ax_tryg.grid(True, linestyle=':', alpha=0.6)
                ax_tryg.legend(loc='upper right')
                fig_tryg.tight_layout()

                save_plot(fig_tryg, f"{folder_name}/wykres_trig_n{n:03d}_m{m:02d}.png")
            else:
                row['tryg_max_err'] = None
                row['tryg_mse_err'] = None

            results.append(row)

        print(f"zakończono obliczenia dla n={n}")

    if results:
        keys = results[0].keys()
        with open(csv_filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            print(f"\nwyniki błędów zapisano do: {csv_filename}")