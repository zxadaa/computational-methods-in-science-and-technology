import numpy as np
import matplotlib.pyplot as plt
import csv
import os


#WZÓR FUNKCJI
def f(x, k=5, m=0.5):
    return np.sin(k * x / np.pi) * np.exp(-m * x / np.pi)


#GENEROWANIE WĘZŁÓW
def generate_uniform_nodes(a, b, n):
    return np.linspace(a, b, n)


def generate_chebyshev_nodes(a, b, n):
    k = np.arange(1, n + 1)
    roots = np.cos((2 * k - 1) / (2 * n) * np.pi)
    return 0.5 * (a + b) + 0.5 * (b - a) * roots


#INTERPOLACJA LAGRANGE'A
def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    y_eval = np.zeros_like(x_eval, dtype=float)
    for k in range(n):
        L_k = np.ones_like(x_eval, dtype=float)
        for i in range(n):
            if i != k:
                L_k *= (x_eval - x_nodes[i]) / (x_nodes[k] - x_nodes[i])
        y_eval += y_nodes[k] * L_k
    return y_eval


#INTERPOLACJA NEWTONA
def divided_difference(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = np.zeros((n, n), dtype=float)
    coef[:, 0] = y_nodes
    for i in range(1, n):
        for j in range(1, i + 1):
            coef[i, j] = (coef[i, j - 1] - coef[i - 1, j - 1]) / (x_nodes[i] - x_nodes[i - j])
    return coef


def newton_interpolation(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    coef = np.diag(divided_difference(x_nodes, y_nodes))
    result = np.full_like(x_eval, coef[n - 1], dtype=float)
    for i in range(n - 2, -1, -1):
        result = result * (x_eval - x_nodes[i]) + coef[i]
    return result


def run_interpolation(func, a, b, n, method='newton', nodes_type='uniform'):
    x_eval = np.linspace(a, b, 1000)
    y_true = func(x_eval)

    #wybór rodzaju węzłów
    if nodes_type == 'uniform':
        x_nodes = generate_uniform_nodes(a, b, n)
    elif nodes_type == 'chebyshev':
        x_nodes = generate_chebyshev_nodes(a, b, n)
    else:
        raise ValueError("niepoprawny typ węzłów")

    y_nodes = func(x_nodes)

    #wybór metody interpolacji
    if method == 'lagrange':
        y_interp = lagrange_interpolation(x_nodes, y_nodes, x_eval)
    elif method == 'newton':
        y_interp = newton_interpolation(x_nodes, y_nodes, x_eval)
    else:
        raise ValueError("niepoprawna metoda")


    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_eval, y_true, 'k--', label='oryginał f(x)')

    line_color = 'r-' if nodes_type == 'uniform' else 'b-'
    ax.plot(x_eval, y_interp, line_color, label=f'Interpolacja ({method})')
    ax.plot(x_nodes, y_nodes, 'ko', label='Węzły')

    ax.set_title(f'Interpolacja: n={n} | Węzły: {nodes_type} | Metoda: {method}')
    ax.set_ylim(np.min(y_true) - 2, np.max(y_true) + 2)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


    return y_true, y_interp, fig


def save_plot(fig, filename):
    fig.savefig(filename, dpi=150)
    print(f"zapisano wykres: {filename}")
    plt.close(fig)


def calculate_max_error(y_true, y_interp):
    return np.max(np.abs(y_true - y_interp))

def calculate_mse_error(y_true, y_interp):
    return np.mean((y_true - y_interp)**2)


if __name__ == "__main__":
    os.makedirs("wykresy", exist_ok=True)

    #parametry zadania
    a_val, b_val = -np.pi * np.pi, np.pi * np.pi

    n_list = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

    results = []

    for n in n_list:
        row = {'n': n}

        y_true, y_interp_uni_lag, fig_uni_lag = run_interpolation(f, a_val, b_val, n, method='lagrange',
                                                                  nodes_type='uniform')
        save_plot(fig_uni_lag, f"wykresy/wykres_n{n:03d}_uniform_lagrange.png")

        _, y_interp_uni_new, fig_uni_new = run_interpolation(f, a_val, b_val, n, method='newton', nodes_type='uniform')
        save_plot(fig_uni_new, f"wykresy/wykres_n{n:03d}_uniform_newton.png")

        row['uni_lag_max'] = calculate_max_error(y_true, y_interp_uni_lag)
        row['uni_lag_mse'] = calculate_mse_error(y_true, y_interp_uni_lag)
        row['uni_new_max'] = calculate_max_error(y_true, y_interp_uni_new)
        row['uni_new_mse'] = calculate_mse_error(y_true, y_interp_uni_new)
        row['diff_uni_lag_new'] = np.max(np.abs(y_interp_uni_lag - y_interp_uni_new))

        _, y_interp_cheb_lag, fig_cheb_lag = run_interpolation(f, a_val, b_val, n, method='lagrange',
                                                               nodes_type='chebyshev')
        save_plot(fig_cheb_lag, f"wykresy/wykres_n{n:03d}_chebyshev_lagrange.png")

        _, y_interp_cheb_new, fig_cheb_new = run_interpolation(f, a_val, b_val, n, method='newton',
                                                               nodes_type='chebyshev')
        save_plot(fig_cheb_new, f"wykresy/wykres_n{n:03d}_chebyshev_newton.png")

        row['cheb_lag_max'] = calculate_max_error(y_true, y_interp_cheb_lag)
        row['cheb_lag_mse'] = calculate_mse_error(y_true, y_interp_cheb_lag)
        row['cheb_new_max'] = calculate_max_error(y_true, y_interp_cheb_new)
        row['cheb_new_mse'] = calculate_mse_error(y_true, y_interp_cheb_new)
        row['diff_cheb_lag_new'] = np.max(np.abs(y_interp_cheb_lag - y_interp_cheb_new))

        results.append(row)
        print(f"zakończono dla n={n}")

    filename = "results.csv"
    if results:
        keys = results[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)