#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, x0, n, count, prob):
    """
    We implement BFGS (Algorithm 6.6 in Algorithms for Optimization, i.e. class textbook)
    to solve the problems in helpers.py. This uses Armijo backtracking line search (only f-calls)
    and then caches the gradient across iterations to save evaluations.
    
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    m = len(x0)
    x = x0.copy()
 
    #Initial evaluation is implemented as follows: only gradient (2 evals). 
    #We skip f(x0) to save 1 eval — set f_val = inf so the first line search point is
    #always accepted via the Armijo condition.
    grad = g(x)
 
    x_best = x.copy()
    f_best = np.inf
    f_val = np.inf
 
    #Inverse Hessian approximation, initialized to identity
    Q = np.eye(m)
 
    #Line search parameters (Armijo backtracking)
    c1 = 1e-4
    rho = 0.5
    max_backtracks = 10
 
    while True:
        #Need at least 1 f-eval (line search) + 2 (gradient) = 3
        if count() + 3 > n:
            break
 
        #Search direction
        d = -Q @ grad
 
        #if d is not a descent direction, then reset Q
        if grad @ d >= 0:
            Q = np.eye(m)
            d = -grad
 
        #Armijo backtracking line search (i.e. f-calls only)
        #Scale initial step so ||alpha * d|| <= 1, preventing huge
        #first steps when the gradient is large (we do this specifically for Rosenbrock).
        #For later iterations, Q \approx H^-1 so ||d|| should already be reasonable enough.
        d_norm = np.linalg.norm(d)
        alpha = min(1.0, 1.0 / d_norm) if d_norm > 1e-12 else 1.0
        directional_deriv = grad @ d  #negative since d is descent
 
        for _ in range(max_backtracks):
            if count() >= n:
                break
            x_new = x + alpha * d
            f_new = f(x_new)  #1 eval
 
            #Track the best point seen so far
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
 
            #Check Armijo sufficient decrease condition
            if f_new <= f_val + c1 * alpha * directional_deriv:
                break
 
            alpha *= rho
 
        #Gradient at new point
        if count() + 2 > n:
            break
 
        grad_new = g(x_new)  #2 evals
 
        #BFGS inverse Hessian update (eq. 6.26 in textbook)
        delta = x_new - x
        gamma = grad_new - grad
        dg = delta @ gamma
 
        if dg > 1e-10:
            Qg = Q @ gamma
            term1 = np.outer(delta, Qg) + np.outer(Qg, delta)
            term2 = (1.0 + (gamma @ Qg) / dg) * np.outer(delta, delta)
            Q = Q - term1 / dg + term2 / dg
 
        #Prepare next iteration (cache gradient)
        x = x_new
        f_val = f_new
        grad = grad_new

    return x_best