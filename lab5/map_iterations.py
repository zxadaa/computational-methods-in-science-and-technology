import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

method = 'newton'
filename = method + '_results.csv'

try:
    df = pd.read_csv(filename, sep=';')
except FileNotFoundError:
    print(f"błąd: nie znaleziono pliku {filename}")
    exit()

df['iteration_count'] = pd.to_numeric(df['iteration_count'], errors='coerce')
df['tolerance_rho'] = pd.to_numeric(df['tolerance_rho'], errors='coerce')

crit_map = {
    "|x_i+1 - x_i| < rho": "Przyrostowe",
    "|f(x_i)| < rho": "Rezydualne"
}

criteria = df['criterion'].unique()
choice = 1
chosen_crit_raw = criteria[choice]

friendly_crit = crit_map.get(chosen_crit_raw, chosen_crit_raw)

subset = df[df['criterion'] == chosen_crit_raw].copy()

method_display = "Siecznych" if method == 'secant' else method.capitalize()

split_indices = []

if method == 'secant':
    pivot_table = subset.pivot_table(
        index=['starting_point_x1', 'starting_point_x0'],
        columns='tolerance_rho',
        values='iteration_count',
        aggfunc=np.mean
    )

    pivot_table = pivot_table.sort_index(level=0)

    x1_values = pivot_table.index.get_level_values('starting_point_x1')
    for i in range(1, len(x1_values)):
        if x1_values[i] != x1_values[i - 1]:
            split_indices.append(i)

    # Zamieniamy etykiety na ostateczny format do wyświetlenia (x0, x1)
    pivot_table.index = [f"({x0}, {x1})" for x1, x0 in pivot_table.index]
    ylabel_text = "Punkty początkowe (x0, x1)"
    fig_size = (10, 14)
else:
    pivot_table = subset.pivot_table(
        index='starting_point_x0',
        columns='tolerance_rho',
        values='iteration_count',
        aggfunc=np.mean
    )
    ylabel_text = "Punkt początkowy x0"
    fig_size = (12, 8)

pivot_table.columns = [f"{c:.0e}" for c in pivot_table.columns]
pivot_table = pivot_table.fillna(0)

plt.figure(figsize=fig_size)

ax = sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu_r",
    cbar_kws={'label': 'Liczba iteracji'}
)

if method == 'secant' and split_indices:
    for idx in split_indices:
        ax.axhline(idx, color='white', linewidth=12)

plt.title(f"Metoda {method_display} - Liczba iteracji\nKryterium: {friendly_crit}", fontsize=14)
plt.xlabel("Dokładność ρ", fontsize=12)
plt.ylabel(ylabel_text, fontsize=12)

output_filename = f'heatmap_iter_{method}_{choice}.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
plt.show()

print(f"wykres zapisany jako: {output_filename}")