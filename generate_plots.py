"""
This file reads the CSV data from plot_data/ and generates plots for the README.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


def read_convergence(filepath):
    iters, fvals = [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iteration"]))
            fvals.append(float(row["f_value"]))
    return np.array(iters), np.array(fvals)


def read_path(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols = [k for k in row if k.startswith("x")]
            rows.append([float(row[c]) for c in sorted(cols)])
    return np.array(rows)


colors = ['#e74c3c', '#2ecc71', '#3498db']
labels = ['Start 1', 'Start 2', 'Start 3']
data_dir = "plot_data"

#Rosenbrock contour + paths
fig, ax = plt.subplots(figsize=(8, 6))

xs = np.linspace(-3, 3, 400)
ys = np.linspace(-2, 4, 400)
X, Y = np.meshgrid(xs, ys)
Z = 100 * (Y - X**2)**2 + (1 - X)**2

levels = np.logspace(-1, 3.5, 25)
ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)

for i in range(3):
    path = read_path(f"{data_dir}/simple1_start{i+1}_path.csv")
    x0 = path[0]
    ax.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], markersize=4,
            linewidth=1.5, label=f'{labels[i]}: ({x0[0]}, {x0[1]})')
    ax.plot(path[0, 0], path[0, 1], 's', color=colors[i], markersize=8, zorder=5)
    ax.plot(path[-1, 0], path[-1, 1], '.', color=colors[i], markersize=14, zorder=5)

ax.plot(1, 1, 'k.', markersize=16, zorder=6, label='Optimum (1, 1)')
ax.set_xlabel('x_1', fontsize=12)
ax.set_ylabel('x_2', fontsize=12)
ax.set_title("BFGS Optimization Paths on Rosenbrock's Function", fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 4)
plt.tight_layout()
plt.savefig('rosenbrock_paths.png', dpi=200)
plt.close()
print("Saved rosenbrock_paths.png")

#Convergence plots (remaining three plots)
prob_titles = {
    "simple1": "Rosenbrock's Function (simple1)",
    "simple2": "Himmelblau's Function (simple2)",
    "simple3": "Powell's Function (simple3)",
}

for prob_name, title in prob_titles.items():
    fig, ax = plt.subplots(figsize=(7, 5))

    for i in range(3):
        filepath = f"{data_dir}/{prob_name}_start{i+1}_convergence.csv"
        iters, fvals = read_convergence(filepath)

        path = read_path(f"{data_dir}/{prob_name}_start{i+1}_path.csv")
        x0_str = np.array2string(path[0], precision=1)

        ax.plot(iters, fvals, 'o-', color=colors[i], markersize=3,
                linewidth=1.5, label=f'{labels[i]}: x_0={x0_str}')

    ax.set_xlabel('Number of iterations', fontsize=12)
    ax.set_ylabel('Function f(x)', fontsize=12)
    ax.set_title(f'Convergence plot for {title}', fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f'{prob_name}_convergence.png'
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved {fname}")