import numpy as np
import matplotlib.pyplot as plt

def f(x, k=5, m=0.5):
    return np.sin(k * x / np.pi) * np.exp(-m * x / np.pi)

def plot_function(f=None, points=None, filename="wykres.png", x_range=(-10, 10)):
    plt.figure(figsize=(8, 5))
    drawn = False

    if callable(f):
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = f(x_vals)
        plt.plot(x_vals, y_vals, 'b-', label="f(x)")
        drawn = True

    if points is not None:
        if isinstance(points, tuple) and len(points) == 2 and hasattr(points[0], '__iter__'):
            x_pts, y_pts = points
        else:
            x_pts = [p[0] for p in points]
            y_pts = [p[1] for p in points]

        plt.plot(x_pts, y_pts, 'ro', label="punkty", markersize=6)
        drawn = True

    if not drawn:
        print("error: nie podano ani funkcji 'f', ani punktów 'points'")
        plt.close()
        return

    #plt.title("wykres f(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"zapisano wykres jako: {filename}")


if __name__ == "__main__":
    plot_function(f=f, filename="wykres_org.png", x_range=(-np.pi * np.pi, np.pi * np.pi))
