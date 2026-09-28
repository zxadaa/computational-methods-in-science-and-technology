import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def generate_cond_plots():
    df1 = pd.read_csv('task1_results.csv', sep=';')
    df2 = pd.read_csv('task2_results.csv', sep=';')

    plt.figure(figsize=(10, 6))

    # Rysowanie linii
    plt.plot(df1['n'], df1['cond_A_float32'], marker='o', markersize=3, linewidth=1.5, color='#e74c3c',
             label='Układ (1) - źle uwarunkowany')
    plt.plot(df2['n'], df2['cond_A_float32'], marker='s', markersize=3, linewidth=1.5, color='#2980b9',
             label='Układ (2) - dobrze uwarunkowany')

    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Współczynnik uwarunkowania cond(A) - skala logarytmiczna', fontsize=12)
    plt.title('Porównanie uwarunkowania macierzy - precyzja float32', fontsize=14)

    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('cond_float32.png', dpi=300)
    print("zapisano wykres: cond_float32.png")
    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(df1['n'], df1['cond_A_float64'], marker='o', markersize=3, linewidth=1.5, color='#e74c3c',
             label='Układ (1) - źle uwarunkowany')
    plt.plot(df2['n'], df2['cond_A_float64'], marker='s', markersize=3, linewidth=1.5, color='#2980b9',
             label='Układ (2) - dobrze uwarunkowany')

    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Współczynnik uwarunkowania cond(A) - skala logarytmiczna ', fontsize=12)
    plt.title('Porównanie uwarunkowania macierzy - precyzja float64', fontsize=14)

    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('cond_float64.png', dpi=300)
    print("zapisano wykres: cond_float64.png")
    plt.close()


generate_cond_plots()