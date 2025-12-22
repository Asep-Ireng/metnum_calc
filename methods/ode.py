"""
ODE (Ordinary Differential Equation) Solvers Module
Implements: Euler, Heun (Improved Euler), Runge-Kutta 4th Order
"""
import numpy as np
import pandas as pd


def euler_method(f, x0: float, y0: float, h: float, x_end: float):
    """
    Euler's Method for solving first-order ODEs.
    dy/dx = f(x, y), y(x0) = y0
    
    Args:
        f: Function f(x, y) where dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value (initial condition)
        h: Step size
        x_end: End point of x
    
    Returns:
        dict with 'x_values', 'y_values', 'table'
    """
    x_values = [x0]
    y_values = [y0]
    
    table_data = [{
        'i': 0,
        'x_i': x0,
        'y_i': y0,
        'f(x_i, y_i)': f(x0, y0),
        'y_{i+1}': y0
    }]
    
    x = x0
    y = y0
    i = 0
    
    while x < x_end - 1e-10:  # Small tolerance for floating point
        i += 1
        f_xy = f(x, y)
        
        # Euler formula: y_{n+1} = y_n + h * f(x_n, y_n)
        y_new = y + h * f_xy
        x_new = x + h
        
        x_values.append(x_new)
        y_values.append(y_new)
        
        table_data.append({
            'i': i,
            'x_i': x_new,
            'y_i': y_new,
            'f(x_i, y_i)': f(x_new, y_new) if x_new < x_end else None,
            'Δy = h·f': h * f_xy
        })
        
        x = x_new
        y = y_new
    
    return {
        'x_values': np.array(x_values),
        'y_values': np.array(y_values),
        'table': pd.DataFrame(table_data),
        'h': h,
        'method': 'Euler'
    }


def heun_method(f, x0: float, y0: float, h: float, x_end: float):
    """
    Heun's Method (Improved Euler / Predictor-Corrector) for solving ODEs.
    
    Args:
        f: Function f(x, y) where dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        x_end: End point of x
    
    Returns:
        dict with 'x_values', 'y_values', 'table'
    """
    x_values = [x0]
    y_values = [y0]
    
    table_data = [{
        'i': 0,
        'x_i': x0,
        'y_i': y0,
        'k1': f(x0, y0),
        'y_pred': y0,
        'k2': f(x0, y0),
        'y_{i+1}': y0
    }]
    
    x = x0
    y = y0
    i = 0
    
    while x < x_end - 1e-10:
        i += 1
        
        # Predictor (Euler step)
        k1 = f(x, y)
        y_pred = y + h * k1
        
        # Corrector
        x_new = x + h
        k2 = f(x_new, y_pred)
        
        # Average slope
        y_new = y + (h / 2) * (k1 + k2)
        
        x_values.append(x_new)
        y_values.append(y_new)
        
        table_data.append({
            'i': i,
            'x_i': x_new,
            'y_i': y_new,
            'k1': k1,
            'y_pred': y_pred,
            'k2': k2,
            'Δy = h(k1+k2)/2': (h / 2) * (k1 + k2)
        })
        
        x = x_new
        y = y_new
    
    return {
        'x_values': np.array(x_values),
        'y_values': np.array(y_values),
        'table': pd.DataFrame(table_data),
        'h': h,
        'method': 'Heun'
    }


def runge_kutta_4(f, x0: float, y0: float, h: float, x_end: float):
    """
    Runge-Kutta 4th Order Method (RK4) for solving ODEs.
    
    Args:
        f: Function f(x, y) where dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        x_end: End point of x
    
    Returns:
        dict with 'x_values', 'y_values', 'table'
    """
    x_values = [x0]
    y_values = [y0]
    
    table_data = [{
        'i': 0,
        'x_i': x0,
        'y_i': y0,
        'k1': None,
        'k2': None,
        'k3': None,
        'k4': None,
        'y_{i+1}': y0
    }]
    
    x = x0
    y = y0
    i = 0
    
    while x < x_end - 1e-10:
        i += 1
        
        # RK4 slopes
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        
        # Weighted average
        y_new = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x_new = x + h
        
        x_values.append(x_new)
        y_values.append(y_new)
        
        table_data.append({
            'i': i,
            'x_i': x_new,
            'y_i': y_new,
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'k4': k4,
            'Δy': (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        })
        
        x = x_new
        y = y_new
    
    return {
        'x_values': np.array(x_values),
        'y_values': np.array(y_values),
        'table': pd.DataFrame(table_data),
        'h': h,
        'method': 'Runge-Kutta 4'
    }


def compare_ode_methods(f, x0: float, y0: float, h: float, x_end: float, 
                         exact_solution=None):
    """
    Compare all ODE methods.
    
    Args:
        f: Function f(x, y)
        x0, y0: Initial conditions
        h: Step size
        x_end: End point
        exact_solution: Optional exact solution function y = g(x)
    
    Returns:
        DataFrame comparing methods
    """
    euler_result = euler_method(f, x0, y0, h, x_end)
    heun_result = heun_method(f, x0, y0, h, x_end)
    rk4_result = runge_kutta_4(f, x0, y0, h, x_end)
    
    # Build comparison table
    x_vals = euler_result['x_values']
    
    comparison = []
    for i, x in enumerate(x_vals):
        row = {
            'x': x,
            'Euler': euler_result['y_values'][i],
            'Heun': heun_result['y_values'][i],
            'RK4': rk4_result['y_values'][i]
        }
        
        if exact_solution is not None:
            exact = exact_solution(x)
            row['Exact'] = exact
            row['Euler Error'] = abs(euler_result['y_values'][i] - exact)
            row['Heun Error'] = abs(heun_result['y_values'][i] - exact)
            row['RK4 Error'] = abs(rk4_result['y_values'][i] - exact)
        
        comparison.append(row)
    
    return {
        'comparison_table': pd.DataFrame(comparison),
        'euler': euler_result,
        'heun': heun_result,
        'rk4': rk4_result
    }
