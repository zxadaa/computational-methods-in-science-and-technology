import math
import csv

#PARAMETRY
n = 15
m = 15
a = -0.5
b = 1.0

#WZÓR FUNKCJI
def f(x):
    return (x - 1) * math.exp(-m * x) + x ** n

def secant_method(x0, x1, rho, criterion, max_iter=100):
    """
    Implementacja metody siecznych
    criterion: 1 - warunek |x_i+1 - x_i| < rho
               2 - warunek |f(x_i)| < rho
    """
    for i in range(max_iter):
        try:
            f_x0 = f(x0)
            f_x1 = f(x1)
        except OverflowError:
            return None, i, "overflow"

        if f_x1 - f_x0 == 0:
            return None, i, "division by zero (f(x1) == f(x0))"

        x_curr = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        if criterion == 1:
            if abs(x_curr - x1) < rho:
                return x_curr, i + 1, "success"
        elif criterion == 2:
            if abs(f(x_curr)) < rho:
                return x_curr, i + 1, "success"

        x0 = x1
        x1 = x_curr

    return x_curr, max_iter, "max iterations"


if __name__ == "__main__":
    reference_root = 0.548182054625515

    step = 0.1
    n_points = int(round((b - a) / step)) + 1
    x0_values = [round(b - i * step, 4) for i in range(n_points)]

    rho_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-11, 1e-15]

    # 1 - warunek |x_i+1 - x_i| < rho, 2 - warunek |f(x_i)| < rho
    criteria = [1, 2]

    results = []

    for crit in criteria:
        criterion_name = "|x_i+1 - x_i| < rho" if crit == 1 else "|f(x_i)| < rho"

        for x0 in x0_values:

            #drugi punkt startowy to a
            if x0 != a:
                for rho in rho_values:
                    root_x, iterations, status = secant_method(x0, a, rho, crit)
                    f_val = f(root_x) if root_x is not None else "N/A"
                    abs_error = abs(root_x - reference_root) if root_x is not None else "N/A"

                    results.append({
                        "criterion": criterion_name,
                        "starting_point_x0": x0,
                        "starting_point_x1": a,
                        "tolerance_rho": rho,
                        "found_root": root_x,
                        "abs_error": abs_error,
                        "function_value_f(x)": f_val,
                        "iteration_count": iterations,
                        "status": status
                    })

            #drugi punkt startowy to b
            if x0 != b:
                for rho in rho_values:
                    root_x, iterations, status = secant_method(x0, b, rho, crit)
                    f_val = f(root_x) if root_x is not None else "N/A"
                    abs_error = abs(root_x - reference_root) if root_x is not None else "N/A"

                    results.append({
                        "criterion": criterion_name,
                        "starting_point_x0": x0,
                        "starting_point_x1": b,
                        "tolerance_rho": rho,
                        "found_root": root_x,
                        "abs_error": abs_error,
                        "function_value_f(x)": f_val,
                        "iteration_count": iterations,
                        "status": status
                    })

    filename = "secant_results.csv"
    columns = [
        "criterion", "starting_point_x0", "starting_point_x1", "tolerance_rho",
        "found_root", "abs_error", "function_value_f(x)", "iteration_count", "status"
    ]

    with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, delimiter=';')
        writer.writeheader()
        writer.writerows(results)

    print(f"wyniki zapisano w: {filename}")