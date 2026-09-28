import numpy as np
import matplotlib.pyplot as plt
import os
import csv


#WZÓR FUNKCJI
def f(x, k=5, m=0.5):
    return np.sin(k * x / np.pi) * np.exp(-m * x / np.pi)


#GENEROWANIE WĘZŁÓW RÓWNOMIERNYCH
def generate_uniform_nodes(a, b, n):
    return np.linspace(a, b, n)


#APROKSYMACJA ŚREDNIOKWADRATOWA - WIELOMIANY ALGEBRAICZNE
def discrete_least_squares_fit(x, y, m, weights=None):
    """
    Funkcja obliczająca współczynniki wielomianu aproksymującego
    metodą najmniejszych kwadratów.
    x, y - punkty danych
    m - stopień wielomianu aproksymującego
    funkcje bazowe: 1,x^2,...,x^m, liczba - m + 1
    """
    n = len(x)
    if weights is None:
        weights = np.ones(n)

    G = np.zeros((m + 1, m + 1))
    B = np.zeros(m + 1)

    for k in range(m + 1):
        for j in range(m + 1):
            G[k, j] = np.sum(weights * (x ** (k + j)))
        B[k] = np.sum(weights * y * (x ** k))

    a_coeffs = np.linalg.solve(G, B)

    def evaluate(x_val):
        x_val = np.asarray(x_val)
        y_val = np.zeros_like(x_val, dtype=float)
        for j, a in enumerate(a_coeffs):
            y_val += a * (x_val ** j)
        return y_val

    return evaluate, a_coeffs


#FUNKCJE POMOCNICZE
def calculate_mse_error(y_true, y_interp):
    return np.mean((y_true - y_interp) ** 2)


def calculate_max_error(y_true, y_interp):
    return np.max(np.abs(y_true - y_interp))


def save_plot(fig, filename):
    fig.savefig(filename, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("wykresy_alg", exist_ok=True)

    #parametry zadania
    a_val, b_val = -np.pi * np.pi, np.pi * np.pi

    n_list = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    m_list = [1, 2, 3, 4, 5, 10, 20, 40, 70]  #rzędy aproksymacji

    x_eval = np.linspace(a_val, b_val, 500)
    y_true = f(x_eval)

    results = []

    for n in n_list:
        x_nodes = generate_uniform_nodes(a_val, b_val, n)
        y_nodes = f(x_nodes)

        for m in m_list:
            row = {'n_nodes': n, 'm_degree': m}

            if m + 1 <= n:
                approx_func_alg, _ = discrete_least_squares_fit(x_nodes, y_nodes, m)
                y_approx_alg = approx_func_alg(x_eval)

                row['alg_max_err'] = calculate_max_error(y_true, y_approx_alg)
                row['alg_mse_err'] = calculate_mse_error(y_true, y_approx_alg)

                fig_alg, ax_alg = plt.subplots(figsize=(8, 5))
                ax_alg.plot(x_eval, y_true, 'k--', label='oryginał f(x)')
                ax_alg.plot(x_eval, y_approx_alg, 'r-', label='aprok. algebraiczna')
                ax_alg.plot(x_nodes, y_nodes, 'ko', markersize=4, label='węzły')

                ax_alg.set_title(f'Aproksymacja algebraiczna: n={n}, m={m}')
                ax_alg.set_ylim(np.min(y_true) - 1, np.max(y_true) + 1)
                ax_alg.grid(True, linestyle=':', alpha=0.6)
                ax_alg.legend(loc='upper right')
                fig_alg.tight_layout()

                save_plot(fig_alg, f"wykresy_alg/wykres_alg_n{n:03d}_m{m:02d}.png")
            else:
                row['alg_max_err'] = None
                row['alg_mse_err'] = None

            results.append(row)

        print(f"zakończono generowanie dla liczby węzłów n={n}")

    filename = "results_alg.csv"
    if results:
        keys = results[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            print(f"\nwyniki błędów zapisano do pliku: {filename}")