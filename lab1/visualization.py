import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('/Users/adriannagrobel/CLionProjects/mownit/results.csv')
except FileNotFoundError:
    print("błąd: nie znaleziono pliku 'results.csv'.")
    exit()


plt.figure(figsize=(10, 6))

plt.plot(df['k'], df['float'], label='float', linewidth=2)
plt.plot(df['k'], df['double'], label='double', linewidth=2)
plt.plot(df['k'], df['long_double'], label='long double', linewidth=2)

plt.xlabel('k')
plt.ylabel('wartość xk')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

ax = plt.gca()


plt.tight_layout()
plt.savefig('wykres.png', dpi=300)
plt.show()