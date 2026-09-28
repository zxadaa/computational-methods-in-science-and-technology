import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results_trig_endpointfalse.csv')

pivot_max = df.pivot(index="n_nodes", columns="m_degree", values="tryg_max_err")
pivot_mse = df.pivot(index="n_nodes", columns="m_degree", values="tryg_mse_err")

log_ticks = [1, 0, -1, -2, -3, -4, -5, -6, -7]
log_labels = ["$10^{1}$", "$10^{0}$", "$10^{-1}$", "$10^{-2}$", "$10^{-3}$", "$10^{-4}$", "$10^{-5}$", "$10^{-6}$", "$10^{-7}$"]

plt.figure(figsize=(12, 9))
sns.heatmap(np.log10(pivot_max.astype(float)),
            annot=pivot_max,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={'ticks': log_ticks})
plt.gca().collections[0].colorbar.set_ticklabels(log_labels)

plt.title("Błąd maksymalny ($E_{max}$)", fontsize=16, pad=25)
plt.xlabel("Stopień wielomianu (m)", fontsize=12, labelpad=10)
plt.ylabel("Liczba węzłów (n)", fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig("heatmap_max_err_endpointfalse.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(np.log10(pivot_mse.astype(float)),
            annot=pivot_mse,
            fmt=".5f",
            cmap="coolwarm",
            cbar_kws={'ticks': log_ticks})
plt.gca().collections[0].colorbar.set_ticklabels(log_labels)

plt.title("Błąd średniokwadratowy (MSE)", fontsize=16, pad=25)
plt.xlabel("Stopień wielomianu (m)", fontsize=12, labelpad=10)
plt.ylabel("Liczba węzłów (n)", fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig("heatmap_mse_endpointfalse.png", dpi=300)
plt.show()