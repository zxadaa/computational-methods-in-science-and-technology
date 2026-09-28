import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def generate_error_plot(input, output):
    df = pd.read_csv(input, sep=';')

    plt.figure(figsize=(10, 6))

    plt.plot(df['n'], df['error_float32'], markersize=2.5, marker='o', linestyle='-', label='precyzja float32', color='#e74c3c')
    plt.plot(df['n'], df['error_float64'], markersize=2.5, marker='s', linestyle='-', label='precyzja float64', color='#2980b9')

    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.xlabel('n')
    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Błąd - skala logarytmiczna', fontsize=12)
    plt.title('Zależność błędu od rozmiaru układu', fontsize=14)

    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"wykres został zapisany jako: {output}")


generate_error_plot('task2_results.csv', "task2_error.png")