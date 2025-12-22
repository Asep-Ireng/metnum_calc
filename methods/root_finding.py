"""
Root Finding Methods Module
Implements: Bisection, Regula Falsi, Newton-Raphson, Secant
"""
import numpy as np
import pandas as pd


def bisection(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Bisection Method for finding roots.
    
    Args:
        f: Function to find root of
        a: Lower bound of interval
        b: Upper bound of interval
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        dict with 'root', 'iterations', 'table' (DataFrame), 'converged'
    """
    # Check if root is bracketed
    if f(a) * f(b) >= 0:
        return {
            'root': None,
            'iterations': 0,
            'table': pd.DataFrame(),
            'converged': False,
            'error': 'f(a) dan f(b) harus memiliki tanda berbeda (akar harus terbracket)'
        }
    
    iterations = []
    c_prev = a
    
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        f_a, f_b, f_c = f(a), f(b), f(c)
        
        # Calculate error
        if i > 1:
            error = abs((c - c_prev) / c) * 100 if c != 0 else 0
        else:
            error = 100.0
        
        iterations.append({
            'Iterasi': i,
            'a': a,
            'b': b,
            'c': c,
            'f(a)': f_a,
            'f(b)': f_b,
            'f(c)': f_c,
            'Error (%)': error
        })
        
        # Check convergence
        if abs(f_c) < tol or error < tol:
            break
        
        # Update interval
        if f_a * f_c < 0:
            b = c
        else:
            a = c
        
        c_prev = c
    
    return {
        'root': c,
        'iterations': i,
        'table': pd.DataFrame(iterations),
        'converged': abs(f(c)) < tol or error < tol,
        'final_error': error
    }


def regula_falsi(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Regula Falsi (False Position) Method for finding roots.
    
    Args:
        f: Function to find root of
        a: Lower bound of interval
        b: Upper bound of interval
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        dict with 'root', 'iterations', 'table' (DataFrame), 'converged'
    """
    if f(a) * f(b) >= 0:
        return {
            'root': None,
            'iterations': 0,
            'table': pd.DataFrame(),
            'converged': False,
            'error': 'f(a) dan f(b) harus memiliki tanda berbeda'
        }
    
    iterations = []
    c_prev = a
    
    for i in range(1, max_iter + 1):
        f_a, f_b = f(a), f(b)
        
        # False position formula
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c)
        
        # Calculate error
        if i > 1:
            error = abs((c - c_prev) / c) * 100 if c != 0 else 0
        else:
            error = 100.0
        
        iterations.append({
            'Iterasi': i,
            'a': a,
            'b': b,
            'c': c,
            'f(a)': f_a,
            'f(b)': f_b,
            'f(c)': f_c,
            'Error (%)': error
        })
        
        # Check convergence
        if abs(f_c) < tol or error < tol:
            break
        
        # Update interval
        if f_a * f_c < 0:
            b = c
        else:
            a = c
        
        c_prev = c
    
    return {
        'root': c,
        'iterations': i,
        'table': pd.DataFrame(iterations),
        'converged': abs(f(c)) < tol or error < tol,
        'final_error': error
    }


def newton_raphson(f, f_prime, x0: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Newton-Raphson Method for finding roots.
    
    Args:
        f: Function to find root of
        f_prime: Derivative of f
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        dict with 'root', 'iterations', 'table' (DataFrame), 'converged'
    """
    iterations = []
    x = x0
    
    for i in range(1, max_iter + 1):
        f_x = f(x)
        f_prime_x = f_prime(x)
        
        if abs(f_prime_x) < 1e-12:
            return {
                'root': None,
                'iterations': i,
                'table': pd.DataFrame(iterations),
                'converged': False,
                'error': 'Turunan mendekati nol, metode gagal'
            }
        
        x_prev = x
        x = x - f_x / f_prime_x
        
        # Calculate error
        if i > 1:
            error = abs((x - x_prev) / x) * 100 if x != 0 else 0
        else:
            error = 100.0
        
        iterations.append({
            'Iterasi': i,
            'x': x,
            'f(x)': f_x,
            "f'(x)": f_prime_x,
            'Error (%)': error
        })
        
        # Check convergence
        if abs(f(x)) < tol or error < tol:
            break
    
    return {
        'root': x,
        'iterations': i,
        'table': pd.DataFrame(iterations),
        'converged': abs(f(x)) < tol or error < tol,
        'final_error': error
    }


def secant(f, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Secant Method for finding roots (no derivative needed).
    
    Args:
        f: Function to find root of
        x0: First initial guess
        x1: Second initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        dict with 'root', 'iterations', 'table' (DataFrame), 'converged'
    """
    iterations = []
    
    for i in range(1, max_iter + 1):
        f_x0 = f(x0)
        f_x1 = f(x1)
        
        if abs(f_x1 - f_x0) < 1e-12:
            return {
                'root': None,
                'iterations': i,
                'table': pd.DataFrame(iterations),
                'converged': False,
                'error': 'Pembagi mendekati nol, metode gagal'
            }
        
        # Secant formula
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        
        # Calculate error
        error = abs((x2 - x1) / x2) * 100 if x2 != 0 else 0
        
        iterations.append({
            'Iterasi': i,
            'x_{n-1}': x0,
            'x_n': x1,
            'x_{n+1}': x2,
            'f(x_{n-1})': f_x0,
            'f(x_n)': f_x1,
            'Error (%)': error
        })
        
        # Check convergence
        if abs(f(x2)) < tol or error < tol:
            break
        
        # Update
        x0 = x1
        x1 = x2
    
    return {
        'root': x2,
        'iterations': i,
        'table': pd.DataFrame(iterations),
        'converged': abs(f(x2)) < tol or error < tol,
        'final_error': error
    }
