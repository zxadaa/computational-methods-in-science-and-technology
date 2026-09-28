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

#FUNKCJE SKLEJANE 3-GO STOPNIA
def create_cubic_spline(x, y, bc_type='natural', f_prime_start=0.0, f_prime_end=0.0):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    A = np.zeros((n, n))
    b_vec = np.zeros(n)

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = delta[i] - delta[i - 1]

    if bc_type == 'natural':
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
    elif bc_type == 'clamped':
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        b_vec[0] = delta[0] - f_prime_start
        A[-1, -2] = h[-1]
        A[-1, -1] = 2 * h[-1]
        b_vec[-1] = f_prime_end - delta[-1]

    sigma = np.linalg.solve(A, b_vec)

    a_coeff = y[:-1]
    b_coeff = delta - h * (sigma[1:] + 2 * sigma[:-1])
    c_coeff = 3 * sigma[:-1]
    d_coeff = (sigma[1:] - sigma[:-1]) / h

    def evaluate(x_val):
        x_val = np.asarray(x_val)
        is_scalar = x_val.ndim == 0
        if is_scalar: x_val = np.array([x_val])

        y_val = np.zeros_like(x_val, dtype=float)
        for idx, xv in np.ndenumerate(x_val):
            i = np.searchsorted(x, xv) - 1
            i = np.clip(i, 0, n - 2)
            dx = xv - x[i]
            y_val[idx] = a_coeff[i] + b_coeff[i] * dx + c_coeff[i] * (dx ** 2) + d_coeff[i] * (dx ** 3)

        return y_val[0] if is_scalar else y_val

    return evaluate


#FUNKCJE SKLEJANE 2-GO STOPNIA
def create_quadratic_spline(x, y, bc_type='zero_curvature', z0_val=0.0):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h

    z = np.zeros(n)

    if bc_type == 'zero_curvature':
        z[0] = delta[0]
    elif bc_type == 'clamped_start':
        z[0] = z0_val

    for i in range(n - 1):
        z[i + 1] = -z[i] + 2 * delta[i]

    a_coeff = y[:-1]
    b_coeff = z[:-1]
    c_coeff = (z[1:] - z[:-1]) / (2 * h)

    def evaluate(x_val):
        x_val = np.asarray(x_val)
        is_scalar = x_val.ndim == 0
        if is_scalar: x_val = np.array([x_val])

        y_val = np.zeros_like(x_val, dtype=float)
        for idx, xv in np.ndenumerate(x_val):
            i = np.searchsorted(x, xv) - 1
            i = np.clip(i, 0, n - 2)
            dx = xv - x[i]
            y_val[idx] = a_coeff[i] + b_coeff[i] * dx + c_coeff[i] * (dx ** 2)

        return y_val[0] if is_scalar else y_val

    return evaluate

def run_interpolation(f, a, b, n, method='cubic_natural', **args):
    x_nodes = generate_uniform_nodes(a, b, n)
    y_nodes = f(x_nodes)

    if method == 'cubic_natural':
        spline = create_cubic_spline(x_nodes, y_nodes, bc_type='natural')
    elif method == 'cubic_clamped':
        fps = args.get('f_prime_start', 0.0)
        fpe = args.get('f_prime_end', 0.0)
        spline = create_cubic_spline(x_nodes, y_nodes, bc_type='clamped', f_prime_start=fps, f_prime_end=fpe)
    elif method == 'quad_zero':
        spline = create_quadratic_spline(x_nodes, y_nodes, bc_type='zero_curvature')
    elif method == 'quad_clamped':
        z0 = args.get('z0_val', 0.0)
        spline = create_quadratic_spline(x_nodes, y_nodes, bc_type='clamped_start', z0_val=z0)
    else:
        raise ValueError("nieznana metoda")

    x_eval = np.linspace(a, b, 500)
    y_true = f(x_eval)
    y_interp = spline(x_eval)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_eval, y_true, 'k--', label='oryginał f(x)')
    ax.plot(x_eval, y_interp, 'r-', label=f'Interpolacja ({method})')
    ax.plot(x_nodes, y_nodes, 'ko', label='Węzły')

    ax.set_title(f'Interpolacja: n={n} | Metoda: {method}')
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

    # parametry zadania
    a_val, b_val = -np.pi * np.pi, np.pi * np.pi
    n_list = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

    results = []

    for n in n_list:
        row = {'n': n}

        #SPLAJN 3-GO STOPNIA (NATURAL)
        y_true, y_interp_uni_cub_nat, fig_uni_cub_nat = run_interpolation(
            f, a_val, b_val, n, method='cubic_natural'
        )
        save_plot(fig_uni_cub_nat, f"wykresy/wykres_n{n:03d}_uniform_cubic_natural.png")

        row['uni_cub_nat_max'] = calculate_max_error(y_true, y_interp_uni_cub_nat)
        row['uni_cub_nat_mse'] = calculate_mse_error(y_true, y_interp_uni_cub_nat)

        #SPLAJN 3-GO STOPNIA (CLAMPED)
        fps, fpe = 0.0, 0.0
        _, y_interp_uni_cub_cla, fig_uni_cub_cla = run_interpolation(
            f, a_val, b_val, n, method='cubic_clamped',
            f_prime_start=fps, f_prime_end=fpe
        )
        save_plot(fig_uni_cub_cla, f"wykresy/wykres_n{n:03d}_uniform_cubic_clamped.png")

        row['uni_cub_cla_max'] = calculate_max_error(y_true, y_interp_uni_cub_cla)
        row['uni_cub_cla_mse'] = calculate_mse_error(y_true, y_interp_uni_cub_cla)

        #SPLAJN 2-GO STOPNIA (ZERO CURVATURE)
        _, y_interp_uni_quad_zer, fig_uni_quad_zer = run_interpolation(
            f, a_val, b_val, n, method='quad_zero'
        )
        save_plot(fig_uni_quad_zer, f"wykresy/wykres_n{n:03d}_uniform_quad_zero.png")

        row['uni_quad_zer_max'] = calculate_max_error(y_true, y_interp_uni_quad_zer)
        row['uni_quad_zer_mse'] = calculate_mse_error(y_true, y_interp_uni_quad_zer)

        #SPLAJN 2-GO STOPNIA (CLAMPED START)
        z0 = 0.0
        _, y_interp_uni_quad_cla, fig_uni_quad_cla = run_interpolation(
            f, a_val, b_val, n, method='quad_clamped', z0_val=z0
        )
        save_plot(fig_uni_quad_cla, f"wykresy/wykres_n{n:03d}_uniform_quad_clamped.png")

        row['uni_quad_cla_max'] = calculate_max_error(y_true, y_interp_uni_quad_cla)
        row['uni_quad_cla_mse'] = calculate_mse_error(y_true, y_interp_uni_quad_cla)

        results.append(row)
        print(f"zakończono generowanie dla n={n}")

    filename = "results_splines.csv"
    if results:
        keys = results[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            print(f"wyniki zapisano do pliku {filename}")