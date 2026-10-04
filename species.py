import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris_data = pd.read_csv('iris_data.csv')

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
pairs = [(features[i], features[j])
         for i in range(len(features))
         for j in range(i + 1, len(features))]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

results = []
for ax, (x_col, y_col) in zip(axes, pairs):
    x = iris_data[x_col].values
    y = iris_data[y_col].values

    # scatter с легендой
    ax.scatter(x, y, alpha=0.7, edgecolor='k', s=40,
               color='blue', label='Данные')

    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    y_pred = k * x + b
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

    ax.plot(xs, k * xs + b, 'r-', linewidth=2,
            label=f'МНК: y = {k:.2f}x + {b:.2f}\nR² = {r2:.3f}')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} от {x_col}')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    results.append((x_col, y_col, k, b, r2))

fig.suptitle('Все попарные зависимости признаков ирисов + прямые МНК',
             fontsize=14, y=1.00)
plt.tight_layout()
plt.show()
