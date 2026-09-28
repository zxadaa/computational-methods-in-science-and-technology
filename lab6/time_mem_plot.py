import pandas as pd
import matplotlib.pyplot as plt


def generate_full_and_zoomed_plots():
    data = pd.read_csv('task3_results.csv', sep=';')

    color_gauss = '#e74c3c'
    color_thomas = '#3498db'

    plt.figure(figsize=(10, 6))
    plt.plot(data['n'], data['time_gauss_float32_s'], marker='o', markersize=5, color=color_gauss, ls='-',
             label='Gauss (float32)')
    plt.plot(data['n'], data['time_gauss_float64_s'], marker='x', markersize=5, color=color_gauss, ls='--',
             label='Gauss (float64)')
    plt.plot(data['n'], data['time_thomas_float32_s'], marker='s', markersize=5, color=color_thomas, ls='-',
             label='Thomas (float32)')
    plt.plot(data['n'], data['time_thomas_float64_s'], marker='^', markersize=5, color=color_thomas, ls='--',
             label='Thomas (float64)')

    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Czas wykonania [s]', fontsize=12)
    plt.title('Czas wykonania algorytmów - pełna skala', fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('task3_time_full.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(data['n'], data['time_gauss_float32_s'], marker='o', markersize=5, color=color_gauss, ls='-',
             label='Gauss (float32)')
    plt.plot(data['n'], data['time_gauss_float64_s'], marker='x', markersize=5, color=color_gauss, ls='--',
             label='Gauss (float64)')
    plt.plot(data['n'], data['time_thomas_float32_s'], marker='s', markersize=5, color=color_thomas, ls='-',
             label='Thomas (float32)')
    plt.plot(data['n'], data['time_thomas_float64_s'], marker='^', markersize=5, color=color_thomas, ls='--',
             label='Thomas (float64)')


    max_thomas_time = max(data['time_thomas_float64_s'].max(), data['time_thomas_float32_s'].max())
    plt.ylim(-max_thomas_time * 0.05, max_thomas_time * 1.2)

    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Czas wykonania [s]', fontsize=12)
    plt.title('Czas wykonania algorytmów - zawężona skala czasu', fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('task3_time_zoom.png', dpi=300)
    plt.close()


    gauss_32_kb = data['mem_gauss_float32_bytes'] / 1024
    gauss_64_kb = data['mem_gauss_float64_bytes'] / 1024
    thomas_32_kb = data['mem_thomas_float32_bytes'] / 1024
    thomas_64_kb = data['mem_thomas_float64_bytes'] / 1024

    plt.figure(figsize=(10, 6))
    plt.plot(data['n'], gauss_32_kb, marker='o', markersize=5, color=color_gauss, ls='-', label='Gauss (float32)')
    plt.plot(data['n'], gauss_64_kb, marker='x', markersize=5, color=color_gauss, ls='--', label='Gauss (float64)')
    plt.plot(data['n'], thomas_32_kb, marker='s', markersize=5, color=color_thomas, ls='-', label='Thomas (float32)')
    plt.plot(data['n'], thomas_64_kb, marker='^', markersize=5, color=color_thomas, ls='--', label='Thomas (float64)')

    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Zajętość pamięci [kB]', fontsize=12)
    plt.title('Zajętość pamięci algorytmów - pełna skala', fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('task3_mem_full.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(data['n'], gauss_32_kb, marker='o', markersize=5, color=color_gauss, ls='-', label='Gauss (float32)')
    plt.plot(data['n'], gauss_64_kb, marker='x', markersize=5, color=color_gauss, ls='--', label='Gauss (float64)')
    plt.plot(data['n'], thomas_32_kb, marker='s', markersize=5, color=color_thomas, ls='-', label='Thomas (float32)')
    plt.plot(data['n'], thomas_64_kb, marker='^', markersize=5, color=color_thomas, ls='--', label='Thomas (float64)')

    max_thomas_mem = max(thomas_64_kb.max(), thomas_32_kb.max())
    plt.ylim(-max_thomas_mem * 0.05, max_thomas_mem * 1.2)

    plt.xlabel('Rozmiar macierzy n', fontsize=12)
    plt.ylabel('Zajętość pamięci [kB]', fontsize=12)
    plt.title('Zajętość pamięci algorytmów - zawężona skala', fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('task3_mem_zoom.png', dpi=300)
    plt.close()



if __name__ == "__main__":
    generate_full_and_zoomed_plots()