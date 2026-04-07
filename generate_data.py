"""
A separate file that runs BFGS with history tracking on each problem and
saves CSV files.
"""
 
import numpy as np
import csv
import os
from project1_py.helpers import Simple1, Simple2, Simple3
 
 
def optimize_with_history(f, g, x0, n, count, prob):
    """
    Same BFGS as project1.py but records x and f at each iteration.
    Uses nolimit mode, so n here is just a large iteration cap.
    """
    m = len(x0)
    x = x0.copy()
    grad = g(x)
 
    x_best = x.copy()
    f_best = np.inf
    f_val = np.inf
 
    Q = np.eye(m)
    c1 = 1e-4
    rho = 0.5
    max_backtracks = 10
 
    x_history = [x.copy()]
    f_history = []
 
    while True:
        if count() + 3 > n:
            break
 
        d = -Q @ grad
        if grad @ d >= 0:
            Q = np.eye(m)
            d = -grad
 
        d_norm = np.linalg.norm(d)
        alpha = min(1.0, 1.0 / d_norm) if d_norm > 1e-12 else 1.0
        directional_deriv = grad @ d
 
        for _ in range(max_backtracks):
            if count() >= n:
                break
            x_new = x + alpha * d
            f_new = f(x_new)
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
            if f_new <= f_val + c1 * alpha * directional_deriv:
                break
            alpha *= rho
 
        x_history.append(x_new.copy())
        f_history.append(f_new)
 
        if count() + 2 > n:
            break
 
        grad_new = g(x_new)
        delta = x_new - x
        gamma = grad_new - grad
        dg = delta @ gamma
 
        if dg > 1e-10:
            Qg = Q @ gamma
            term1 = np.outer(delta, Qg) + np.outer(Qg, delta)
            term2 = (1.0 + (gamma @ Qg) / dg) * np.outer(delta, delta)
            Q = Q - term1 / dg + term2 / dg
 
        x = x_new
        f_val = f_new
        grad = grad_new
 
    return x_best, x_history, f_history

starts_2d = [
    np.array([-2.0,  2.0]),
    np.array([ 2.0, -1.0]),
    np.array([-1.5, -0.5]),
]
starts_4d = [
    np.array([ 1.0, -1.0,  2.0, -2.0]),
    np.array([-2.0,  2.0, -1.0,  1.0]),
    np.array([ 0.5,  0.5, -1.5,  1.5]),
]
 
problems = [
    ("simple1", Simple1, starts_2d),
    ("simple2", Simple2, starts_2d),
    ("simple3", Simple3, starts_4d),
]
 
iter_caps = {"simple1": 50, "simple2": 50, "simple3": 80}
 
os.makedirs("plot_data", exist_ok=True)
 
#Generate and save all data to a folder
for prob_name, ProbClass, starts in problems:
    max_iters = iter_caps[prob_name]
 
    for si, x0 in enumerate(starts):
        p = ProbClass()
        p.nolimit()
        _, x_hist, f_hist = optimize_with_history(
            p.f, p.g, x0, max_iters * 5, p.count, p.prob
        )
 
        #Trim to cap
        x_hist = x_hist[:max_iters + 1]  # includes x0
        f_hist = f_hist[:max_iters]
 
        #Save convergence data: iteration, f_value
        conv_file = f"plot_data/{prob_name}_start{si+1}_convergence.csv"
        with open(conv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["iteration", "f_value"])
            for it, fv in enumerate(f_hist, start=1):
                writer.writerow([it, fv])
 
        #Save path data: iteration, x0, x1, ... (for contour plot)
        path_file = f"plot_data/{prob_name}_start{si+1}_path.csv"
        xdim = len(x0)
        with open(path_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["iteration"] + [f"x{j}" for j in range(xdim)]
            writer.writerow(header)
            for it, xv in enumerate(x_hist):
                writer.writerow([it] + list(xv))
 
        print(f"Saved {conv_file} ({len(f_hist)} iters) and {path_file} ({len(x_hist)} pts)")
 
print("\nAll data saved to plot_data/")