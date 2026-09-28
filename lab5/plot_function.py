import numpy as np
import matplotlib.pyplot as plt

#WZÓR FUNKCJI
def f(x, n=15, m=15):
    return (x - 1) * np.exp(-m * x) + x ** n

def plot_functions(filename1="wykres_pelny.png", filename2="wykres_zblizenie.png"):
    x_full = np.linspace(-0.5, 1.0, 1000)
    y_full = f(x_full)
    x_zero = 0.548182

    plt.figure(figsize=(8, 5))
    plt.plot(x_full, y_full, 'b-', linewidth=2)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.plot(x_zero, 0, 'ro')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename1, dpi=150)
    plt.close()
    print(f"zapisano wykres pełny jako: {filename1}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_full, y_full, 'b-', linewidth=2)
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1)
    plt.plot(x_zero, 0, 'ro')

    plt.xlim(-1.0, 1.0)
    plt.ylim(-2, 2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename2, dpi=150)
    plt.close()
    print(f"zapisano wykres zbliżony jako: {filename2}")


if __name__ == "__main__":
    plot_functions()