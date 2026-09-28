import csv
import numpy as np
import matplotlib.pyplot as plt

def euler(f, u0, t0, t_end, h):
    """
    rozwiązuje równanie różniczkowe du/dt = f(t, u) metodą Eulera,
    parametry:
    f: funkcja f(t, u) opisująca równanie
    u0: warunek początkowy u(t0) = u0
    t0: czas początkowy
    t_end: zas końcowy
    h: krok czasowy
    """
    t_values = [t0]
    u_values = [u0]

    t = t0
    u = u0

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        u_next = u + h * f(t, u)
        t = t + h

        t_values.append(t)
        u_values.append(u_next)
        u = u_next

    return t_values, u_values


def runge_kutta_4(f, u0, t0, t_end, h):
    """
    rozwiązuje równanie różniczkowe du/dt = f(t, u) metodą Rungego-Kutty 4. rzędu,
    parametry:
    f: funkcja f(t, u) opisująca równanie
    u0: warunek początkowy u(t0) = u0
    t0: czas początkowy
    t_end: zas końcowy
    h: krok czasowy
    """
    t_values = [t0]
    u_values = [u0]

    t = t0
    u = u0

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        #obliczanie 4 współczynników k
        k1 = f(t, u)
        k2 = f(t + 0.5 * h, u + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, u + 0.5 * h * k2)
        k4 = f(t + h, u + h * k3)

        #główny krok metody RK4
        u_next = u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h

        t_values.append(t)
        u_values.append(u_next)
        u = u_next

    return t_values, u_values


x0 = np.pi / 4
x1 = 3 * np.pi
m = 2
k = 1


def exact_solution(x):
    return np.exp(-k * np.cos(m * x)) - k * np.cos(m * x) + 1


#y' = k*m*y*sin(m*x) + k^2*m*sin(m*x)*cos(m*x)
def f_ode(x, y):
    return k * m * y * np.sin(m * x) + (k ** 2) * m * np.sin(m * x) * np.cos(m * x)


#wyliczenie parametru a
a = exact_solution(x0)
print(f"warunek początkowy: a = {a:.4f}")

h_values = [0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001]

x_exact_plot = np.linspace(x0, x1, 1000)
y_exact_plot = exact_solution(x_exact_plot)

filename_results = "results.csv"
filename_errors = "errors.csv"


errors_summary = []

with open(filename_results, mode='w', newline='', encoding='utf-8') as f_res:
    writer_res = csv.writer(f_res)
    writer_res.writerow(["h", "x", "y_euler", "y_rk4", "y_exact"])

    for h in h_values:
        x_euler, y_euler = euler(f_ode, a, x0, x1, h)
        x_rk4, y_rk4 = runge_kutta_4(f_ode, a, x0, x1, h)

        y_exact_nodes = exact_solution(np.array(x_euler))

        max_err_euler = np.max(np.abs(np.array(y_euler) - y_exact_nodes))
        max_err_rk4 = np.max(np.abs(np.array(y_rk4) - y_exact_nodes))

        print(f"przetworzono krok h = {h:.0e}")

        errors_summary.append([h, max_err_euler, max_err_rk4])

        for x_val, y_eu, y_rk, y_ex in zip(x_euler, y_euler, y_rk4, y_exact_nodes):
            writer_res.writerow([h, x_val, y_eu, y_rk, y_ex])

        plt.figure(figsize=(10, 6))
        plt.plot(x_exact_plot, y_exact_plot, 'k-', linewidth=2, label='Rozwiązanie dokładne')
        plt.plot(x_euler, y_euler, 'r--', linewidth=1.5, label='Metoda Eulera')
        plt.plot(x_rk4, y_rk4, 'b-.', linewidth=1.5, label='Metoda RK4')
        plt.title(f'Porównanie metod dla h = {h:.0e}')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()

        output_image = f"wykres_h_{h:.0e}.png"
        plt.savefig(output_image, dpi=300)
        plt.close()

with open(filename_errors, mode='w', newline='', encoding='utf-8') as f_err:
    writer_err = csv.writer(f_err)
    writer_err.writerow(["h", "max_err_euler", "max_err_rk4"])
    writer_err.writerows(errors_summary)

print(f"wyniki zapisano w pliku: {filename_results}")
print(f"maksymalne błędy zapisano w pliku: {filename_errors}")