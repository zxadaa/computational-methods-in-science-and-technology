import math
import csv

#PARAMETRY ZADANIA
n = 15
m = 15
a = -0.5
b = 1.0

#WZÓR FUNKCJI
def f(x):
    return (x - 1) * math.exp(-m * x) + x ** n

#POCHODNA FUNKCJI
def df(x):
    return math.exp(-m * x) - m * (x - 1) * math.exp(-m * x) + n * x ** (n - 1)


def newton_method(x0, rho, criterion, max_iter=100):
    """
    Implementacja metody Newtona-Raphsona
    criterion: 1 - warunek |x_i+1 - x_i| < rho
               2 - warunek |f(x_i)| < rho
    """
    x_prev = x0
    for i in range(max_iter):
        try:
            fx = f(x_prev)
            dfx = df(x_prev)
        except OverflowError:
            return None, i, "overflow"

        if dfx == 0:
            return None, i, "dfx = 0"

        x_curr = x_prev - (fx / dfx)

        if criterion == 1:
            if abs(x_curr - x_prev) < rho:
                return x_curr, i + 1, "success"
        elif criterion == 2:
            if abs(f(x_curr)) < rho:
                return x_curr, i + 1, "success"

        x_prev = x_curr

    return x_curr, max_iter, "max iterations"


if __name__ == "__main__":
    reference_root = 0.548182054625515

    step = 0.1
    n_points = int(round((b - a) / step)) + 1
    x0_values = [round(a + i * step, 4) for i in range(n_points)]

    rho_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-11, 1e-15]
    criteria = [1, 2]

    results = []

    for crit in criteria:
        criterion_name = "|x_i+1 - x_i| < rho" if crit == 1 else "|f(x_i)| < rho"

        for x0 in x0_values:
            for rho in rho_values:
                root_x, iterations, status = newton_method(x0, rho, crit)

                if root_x is not None:
                    f_val = f(root_x)
                    abs_error = abs(root_x - reference_root)
                else:
                    f_val = "N/A"
                    abs_error = "N/A"

                results.append({
                    "criterion": criterion_name,
                    "starting_point_x0": x0,
                    "tolerance_rho": rho,
                    "found_root": root_x,
                    "abs_error": abs_error,
                    "function_value_f(x)": f_val,
                    "iteration_count": iterations,
                    "status": status
                })

    filename = "newton_results.csv"
    columns = [
        "criterion", "starting_point_x0", "tolerance_rho",
        "found_root", "abs_error", "function_value_f(x)", "iteration_count", "status"
    ]

    with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, delimiter=';')
        writer.writeheader()
        writer.writerows(results)

    print(f"wyniki zapisano w: {filename}")