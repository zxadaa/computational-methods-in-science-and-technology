import numpy as np
import matplotlib.pyplot as plt
import csv
import os


#WZÓR FUNKCJI I JEJ POCHODNEJ
def f(x, k=5, m=0.5):
    return np.sin(k * x / np.pi) * np.exp(-m * x / np.pi)


def df(x, k=5, m=0.5):
    term1 = (k / np.pi) * np.cos(k * x / np.pi) * np.exp(-m * x / np.pi)
    term2 = np.sin(k * x / np.pi) * (-m / np.pi) * np.exp(-m * x / np.pi)
    return term1 + term2


#GENEROWANIE WĘZŁÓW
def generate_uniform_nodes(a, b, n):
    return np.linspace(a, b, n)


def generate_chebyshev_nodes(a, b, n):
    k = np.arange(1, n + 1)
    roots = np.cos((2 * k - 1) / (2 * n) * np.pi)
    return 0.5 * (a + b) + 0.5 * (b - a) * roots


#INTERPOLACJA HERMITE'A
def get_hermite_coefficients(x_nodes, y_nodes, dy_nodes):
    n = len(x_nodes)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

    for i in range(n):
        z[2 * i] = x_nodes[i]
        z[2 * i + 1] = x_nodes[i]

        Q[2 * i, 0] = y_nodes[i]
        Q[2 * i + 1, 0] = y_nodes[i]
        Q[2 * i + 1, 1] = dy_nodes[i]

        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])

    return z, np.diag(Q)


def hermite_interpolation(x_nodes, y_nodes, dy_nodes, x_eval):
    z, coef = get_hermite_coefficients(x_nodes, y_nodes, dy_nodes)
    n = len(coef)
    result = np.full_like(x_eval, coef[n - 1], dtype=float)

    for i in range(n - 2, -1, -1):
        result = result * (x_eval - z[i]) + coef[i]

    return result


#FUNKCJA GŁÓWNA URUCHAMIAJĄCA INTERPOLACJĘ
def run_and_plot_interpolation(func, dfunc, a, b, n, nodes_type='uniform'):
    x_eval = np.linspace(a, b, 1000)
    y_true = func(x_eval)

    if nodes_type == 'uniform':
        x_nodes = generate_uniform_nodes(a, b, n)
    elif nodes_type == 'chebyshev':
        x_nodes = generate_chebyshev_nodes(a, b, n)
    else:
        raise ValueError("niepoprawny typ węzłów")

    y_nodes = func(x_nodes)
    dy_nodes = dfunc(x_nodes)

    y_interp = hermite_interpolation(x_nodes, y_nodes, dy_nodes, x_eval)


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_eval, y_true, 'k--', label='Oryginał f(x)')

    line_color = 'r-' if nodes_type == 'uniform' else 'b-'
    ax.plot(x_eval, y_interp, line_color, label='Interpolacja (Hermite)')
    ax.plot(x_nodes, y_nodes, 'ko', label='Węzły')

    ax.set_title(f'Interpolacja Hermite: n={n} | Węzły: {nodes_type}')

    y_min, y_max = np.min(y_true), np.max(y_true)
    y_margin = (y_max - y_min) * 0.5
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.grid(True)
    ax.legend(loc='upper right')
    fig.tight_layout()

    return y_true, y_interp, fig


#FUNKCJE POMOCNICZE
def save_plot(fig, filename):
    fig.savefig(filename, dpi=150)
    print(f"zapisano wykres: {filename}")
    plt.close(fig)


def calculate_max_error(y_true, y_interp):
    return np.max(np.abs(y_true - y_interp))


def calculate_mse_error(y_true, y_interp):
    return np.mean((y_true - y_interp) ** 2)


if __name__ == "__main__":
    os.makedirs("wykresy", exist_ok=True)

    #parametry zadania
    a_val, b_val = -np.pi * np.pi, np.pi * np.pi
    n_list = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

    results = []

    for n in n_list:
        row = {'n': n}

        y_true, y_interp_uni, fig_uni = run_and_plot_interpolation(f, df, a_val, b_val, n, nodes_type='uniform')
        save_plot(fig_uni, f"wykresy/wykres_n{n:02d}_uniform_hermite.png")

        row['uni_max_err'] = calculate_max_error(y_true, y_interp_uni)
        row['uni_mse_err'] = calculate_mse_error(y_true, y_interp_uni)

        _, y_interp_cheb, fig_cheb = run_and_plot_interpolation(f, df, a_val, b_val, n, nodes_type='chebyshev')
        save_plot(fig_cheb, f"wykresy/wykres_n{n:02d}_chebyshev_hermite.png")

        row['cheb_max_err'] = calculate_max_error(y_true, y_interp_cheb)
        row['cheb_mse_err'] = calculate_mse_error(y_true, y_interp_cheb)

        results.append(row)
        print(f"zakończono dla n={n}")

    filename = "results_hermite.csv"
    if results:
        keys = results[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            print(f"\nwyniki błędów zapisano do pliku: {filename}")