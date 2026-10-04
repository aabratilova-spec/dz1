import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N_steps = 1000

steps = np.random.choice([-1, 1], size=N_steps)
x = np.concatenate([[0], np.cumsum(steps)])
n = np.arange(len(x))

fig1, ax = plt.subplots(figsize=(11, 5))
ax.plot(n, x, color='navy', linewidth=1.2,
        label=f'траектория частицы (N = {N_steps})')
ax.fill_between(n, -np.sqrt(n), np.sqrt(n), alpha=0.15, color='red',
                label=r'коридор $\pm\sqrt{N}$ (1σ)')
ax.fill_between(n, -2*np.sqrt(n), 2*np.sqrt(n), alpha=0.08, color='orange',
                label=r'коридор $\pm 2\sqrt{N}$ (2σ)')
ax.axhline(0, color='black', linewidth=0.6)
ax.set_xlabel('N (номер шага)')
ax.set_ylabel('x (положение)')
ax.set_title('Случайное блуждание: траектория одной частицы')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

n_particles = 1000
N_steps = 1000

steps = np.random.choice([-1, 1], size=(n_particles, N_steps))
positions = steps.sum(axis=1)

fig2, ax = plt.subplots(figsize=(10, 5))
ax.hist(positions, bins=35, density=True, alpha=0.7,
        edgecolor='black', color='steelblue',
        label=f'{n_particles} частиц после N = {N_steps} шагов')

sigma_theory = np.sqrt(N_steps)
xs = np.linspace(positions.min(), positions.max(), 300)
pdf = np.exp(-xs**2 / (2 * sigma_theory**2)) / (sigma_theory * np.sqrt(2 * np.pi))
ax.plot(xs, pdf, 'r-', linewidth=2.5,
        label=f'теория N(0, √N),  √N = {sigma_theory:.1f}')

ax.axvline(positions.mean(), color='green', linestyle='--', linewidth=2,
           label=f'среднее = {positions.mean():.2f}')
ax.axvline( positions.std(), color='purple', linestyle=':', linewidth=2,
            label=f'±σ эксп. = ±{positions.std():.2f}')
ax.axvline(-positions.std(), color='purple', linestyle=':', linewidth=2)

ax.set_xlabel('x (положение после 1000 шагов)')
ax.set_ylabel('плотность вероятности')
ax.set_title('Распределение положений 1000 частиц после 1000 шагов')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Среднее (эксперимент):  {positions.mean():.3f}')
print(f'σ (эксперимент):        {positions.std():.3f}')
print(f'√N (теория):            {np.sqrt(N_steps):.3f}')

n_particles = 2000
N_max = 1000
Ns = np.arange(10, N_max + 1, 20)

steps = np.random.choice([-1, 1], size=(n_particles, N_max))
trajectories = np.cumsum(steps, axis=1)

sigmas = np.array([trajectories[:, N - 1].std() for N in Ns])

# степенная аппроксимация σ = c · N^α
alpha, log_c = np.polyfit(np.log(Ns), np.log(sigmas), 1)
c = np.exp(log_c)
N_fit = np.linspace(Ns.min(), Ns.max(), 300)

fig3, ax = plt.subplots(figsize=(10, 5))
ax.plot(Ns, sigmas, 'o', markersize=6, color='steelblue',
        label='эксперимент: std(x) по частицам')
ax.plot(N_fit, np.sqrt(N_fit), 'r-', linewidth=2.5,
        label=r'теория: $\sigma = \sqrt{N}$')
ax.plot(N_fit, c * N_fit**alpha, 'g--', linewidth=2,
        label=f'МНК: σ = {c:.3f}·N^{alpha:.3f}')

ax.set_xlabel('N (число шагов)')
ax.set_ylabel('σ (стандартное отклонение x)')
ax.set_title('Зависимость σ от N: подтверждение закона √N')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'\nОценка показателя степени α = {alpha:.4f}  (теория: 0.5)')