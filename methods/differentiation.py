"""
Numerical Differentiation Methods Module
Implements: Forward Difference, Backward Difference, Central Difference
"""
import numpy as np
import pandas as pd


def forward_difference(f, x: float, h: float = 0.1):
    """
    Forward Difference Approximation of first derivative.
    f'(x) ≈ [f(x+h) - f(x)] / h
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size
    
    Returns:
        dict with 'derivative', 'f_x', 'f_x_plus_h', 'h'
    """
    f_x = f(x)
    f_x_plus_h = f(x + h)
    
    derivative = (f_x_plus_h - f_x) / h
    
    return {
        'derivative': derivative,
        'f(x)': f_x,
        'f(x+h)': f_x_plus_h,
        'h': h,
        'formula': "f'(x) ≈ [f(x+h) - f(x)] / h",
        'order': 'O(h) - First Order'
    }


def backward_difference(f, x: float, h: float = 0.1):
    """
    Backward Difference Approximation of first derivative.
    f'(x) ≈ [f(x) - f(x-h)] / h
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size
    
    Returns:
        dict with 'derivative', 'f_x', 'f_x_minus_h', 'h'
    """
    f_x = f(x)
    f_x_minus_h = f(x - h)
    
    derivative = (f_x - f_x_minus_h) / h
    
    return {
        'derivative': derivative,
        'f(x)': f_x,
        'f(x-h)': f_x_minus_h,
        'h': h,
        'formula': "f'(x) ≈ [f(x) - f(x-h)] / h",
        'order': 'O(h) - First Order'
    }


def central_difference(f, x: float, h: float = 0.1):
    """
    Central Difference Approximation of first derivative.
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate derivative
        h: Step size
    
    Returns:
        dict with 'derivative', 'f_x_plus_h', 'f_x_minus_h', 'h'
    """
    f_x_plus_h = f(x + h)
    f_x_minus_h = f(x - h)
    
    derivative = (f_x_plus_h - f_x_minus_h) / (2 * h)
    
    return {
        'derivative': derivative,
        'f(x+h)': f_x_plus_h,
        'f(x-h)': f_x_minus_h,
        'h': h,
        'formula': "f'(x) ≈ [f(x+h) - f(x-h)] / (2h)",
        'order': 'O(h²) - Second Order'
    }


def second_derivative_central(f, x: float, h: float = 0.1):
    """
    Central Difference Approximation of second derivative.
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate second derivative
        h: Step size
    
    Returns:
        dict with 'derivative', values, 'h'
    """
    f_x = f(x)
    f_x_plus_h = f(x + h)
    f_x_minus_h = f(x - h)
    
    derivative = (f_x_plus_h - 2 * f_x + f_x_minus_h) / (h ** 2)
    
    return {
        'derivative': derivative,
        'f(x)': f_x,
        'f(x+h)': f_x_plus_h,
        'f(x-h)': f_x_minus_h,
        'h': h,
        'formula': "f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²",
        'order': 'O(h²) - Second Order'
    }


def compare_differentiation_methods(f, x: float, exact_derivative: float = None):
    """
    Compare all differentiation methods with different step sizes.
    
    Args:
        f: Function to differentiate
        x: Point to evaluate
        exact_derivative: Exact derivative value for error calculation
    
    Returns:
        DataFrame comparing methods and step sizes
    """
    h_values = [1, 0.5, 0.1, 0.05, 0.01]
    results = []
    
    for h in h_values:
        # Forward
        fwd = forward_difference(f, x, h)
        row = {
            'Metode': 'Forward',
            'h': h,
            'Hasil': fwd['derivative']
        }
        if exact_derivative is not None:
            row['Error (%)'] = abs((fwd['derivative'] - exact_derivative) / exact_derivative) * 100
        results.append(row)
        
        # Backward
        bwd = backward_difference(f, x, h)
        row = {
            'Metode': 'Backward',
            'h': h,
            'Hasil': bwd['derivative']
        }
        if exact_derivative is not None:
            row['Error (%)'] = abs((bwd['derivative'] - exact_derivative) / exact_derivative) * 100
        results.append(row)
        
        # Central
        ctr = central_difference(f, x, h)
        row = {
            'Metode': 'Central',
            'h': h,
            'Hasil': ctr['derivative']
        }
        if exact_derivative is not None:
            row['Error (%)'] = abs((ctr['derivative'] - exact_derivative) / exact_derivative) * 100
        results.append(row)
    
    return pd.DataFrame(results)


def differentiation_table(f, x: float, h_values: list = None, exact: float = None):
    """
    Create a comprehensive table of differentiation results.
    
    Args:
        f: Function to differentiate
        x: Point to evaluate
        h_values: List of step sizes
        exact: Exact derivative value
    
    Returns:
        DataFrame with results for all methods and step sizes
    """
    if h_values is None:
        h_values = [1, 0.5, 0.1, 0.05, 0.01]
    
    results = []
    
    for h in h_values:
        fwd = forward_difference(f, x, h)['derivative']
        bwd = backward_difference(f, x, h)['derivative']
        ctr = central_difference(f, x, h)['derivative']
        
        row = {
            'h': h,
            'Forward': fwd,
            'Backward': bwd,
            'Central': ctr
        }
        
        if exact is not None:
            row['Forward Error (%)'] = abs((fwd - exact) / exact) * 100
            row['Backward Error (%)'] = abs((bwd - exact) / exact) * 100
            row['Central Error (%)'] = abs((ctr - exact) / exact) * 100
        
        results.append(row)
    
    return pd.DataFrame(results)
