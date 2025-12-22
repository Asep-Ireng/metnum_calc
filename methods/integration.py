"""
Numerical Integration Methods Module
Implements: Trapezoidal, Simpson's 1/3, Simpson's 3/8
"""
import numpy as np
import pandas as pd


def trapezoidal(f, a: float, b: float, n: int = 10):
    """
    Trapezoidal Rule for numerical integration.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of sub-intervals
    
    Returns:
        dict with 'integral', 'table', 'h', 'n'
    """
    h = (b - a) / n
    
    table_data = []
    total_sum = 0
    
    for i in range(n + 1):
        x_i = a + i * h
        f_i = f(x_i)
        
        # Coefficient: 1 for endpoints, 2 for middle points
        if i == 0 or i == n:
            coef = 1
            coef_str = "1 (ujung)"
        else:
            coef = 2
            coef_str = "2 (tengah)"
        
        contribution = coef * f_i
        total_sum += contribution
        
        table_data.append({
            'i': i,
            'x_i': x_i,
            'f(x_i)': f_i,
            'Koefisien': coef_str,
            'Kontribusi': contribution
        })
    
    integral = (h / 2) * total_sum
    
    return {
        'integral': integral,
        'table': pd.DataFrame(table_data),
        'h': h,
        'n': n,
        'sum': total_sum
    }


def simpson_13(f, a: float, b: float, n: int = 10):
    """
    Simpson's 1/3 Rule for numerical integration.
    Requires n to be even.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of sub-intervals (must be even)
    
    Returns:
        dict with 'integral', 'table', 'h', 'n', 'error' if n is odd
    """
    if n % 2 != 0:
        return {
            'integral': None,
            'table': pd.DataFrame(),
            'h': None,
            'n': n,
            'error': 'n harus genap untuk Simpson 1/3'
        }
    
    h = (b - a) / n
    
    table_data = []
    total_sum = 0
    
    for i in range(n + 1):
        x_i = a + i * h
        f_i = f(x_i)
        
        # Coefficient pattern: 1, 4, 2, 4, 2, ..., 4, 1
        if i == 0 or i == n:
            coef = 1
            coef_str = "1 (ujung)"
        elif i % 2 == 1:
            coef = 4
            coef_str = "4 (ganjil)"
        else:
            coef = 2
            coef_str = "2 (genap)"
        
        contribution = coef * f_i
        total_sum += contribution
        
        table_data.append({
            'i': i,
            'x_i': x_i,
            'f(x_i)': f_i,
            'Koefisien': coef_str,
            'Kontribusi': contribution
        })
    
    integral = (h / 3) * total_sum
    
    return {
        'integral': integral,
        'table': pd.DataFrame(table_data),
        'h': h,
        'n': n,
        'sum': total_sum
    }


def simpson_38(f, a: float, b: float, n: int = 9):
    """
    Simpson's 3/8 Rule for numerical integration.
    Requires n to be divisible by 3.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of sub-intervals (must be divisible by 3)
    
    Returns:
        dict with 'integral', 'table', 'h', 'n', 'error' if n not divisible by 3
    """
    if n % 3 != 0:
        return {
            'integral': None,
            'table': pd.DataFrame(),
            'h': None,
            'n': n,
            'error': 'n harus kelipatan 3 untuk Simpson 3/8'
        }
    
    h = (b - a) / n
    
    table_data = []
    total_sum = 0
    
    for i in range(n + 1):
        x_i = a + i * h
        f_i = f(x_i)
        
        # Coefficient pattern: 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1
        if i == 0 or i == n:
            coef = 1
            coef_str = "1 (ujung)"
        elif i % 3 == 0:
            coef = 2
            coef_str = "2 (kelipatan 3)"
        else:
            coef = 3
            coef_str = "3 (lainnya)"
        
        contribution = coef * f_i
        total_sum += contribution
        
        table_data.append({
            'i': i,
            'x_i': x_i,
            'f(x_i)': f_i,
            'Koefisien': coef_str,
            'Kontribusi': contribution
        })
    
    integral = (3 * h / 8) * total_sum
    
    return {
        'integral': integral,
        'table': pd.DataFrame(table_data),
        'h': h,
        'n': n,
        'sum': total_sum
    }


def compare_integration_methods(f, a: float, b: float, exact_value: float = None):
    """
    Compare all integration methods with different step sizes.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        exact_value: Exact integral value for error calculation (optional)
    
    Returns:
        DataFrame comparing methods and step sizes
    """
    results = []
    
    # Test different n values
    n_values = [4, 8, 16, 32]
    
    for n in n_values:
        # Trapezoidal
        trap = trapezoidal(f, a, b, n)
        if trap['integral'] is not None:
            row = {
                'Metode': 'Trapezoidal',
                'n': n,
                'h': trap['h'],
                'Hasil': trap['integral']
            }
            if exact_value:
                row['Error (%)'] = abs((trap['integral'] - exact_value) / exact_value) * 100
            results.append(row)
        
        # Simpson 1/3 (n must be even)
        if n % 2 == 0:
            simp13 = simpson_13(f, a, b, n)
            if simp13['integral'] is not None:
                row = {
                    'Metode': 'Simpson 1/3',
                    'n': n,
                    'h': simp13['h'],
                    'Hasil': simp13['integral']
                }
                if exact_value:
                    row['Error (%)'] = abs((simp13['integral'] - exact_value) / exact_value) * 100
                results.append(row)
        
        # Simpson 3/8 (n must be divisible by 3)
        n_38 = n if n % 3 == 0 else (n // 3 + 1) * 3
        simp38 = simpson_38(f, a, b, n_38)
        if simp38['integral'] is not None:
            row = {
                'Metode': 'Simpson 3/8',
                'n': n_38,
                'h': simp38['h'],
                'Hasil': simp38['integral']
            }
            if exact_value:
                row['Error (%)'] = abs((simp38['integral'] - exact_value) / exact_value) * 100
            results.append(row)
    
    return pd.DataFrame(results)
