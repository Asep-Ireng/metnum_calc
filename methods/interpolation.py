"""
Interpolation Methods Module
Implements: Newton Divided Differences, Lagrange Interpolation
"""
import numpy as np
import pandas as pd


def newton_divided_difference(x_points: np.ndarray, y_points: np.ndarray, x_target: float = None):
    """
    Newton's Divided Difference Interpolation.
    
    Args:
        x_points: Array of x values (data points)
        y_points: Array of y values (data points)
        x_target: Point to interpolate (optional)
    
    Returns:
        dict with 'coefficients', 'polynomial_str', 'table', 'interpolated_value'
    """
    n = len(x_points)
    
    # Create divided difference table
    # dd[i][j] represents the j-th divided difference at point i
    dd = np.zeros((n, n))
    dd[:, 0] = y_points
    
    # Build divided difference table
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (x_points[i + j] - x_points[i])
    
    # Create table DataFrame for display
    columns = ['x', 'f[x]'] + [f'f[x{",x" * i}]' for i in range(1, n)]
    table_data = []
    for i in range(n):
        row = [x_points[i]]
        for j in range(n):
            if j <= n - i - 1:
                row.append(dd[i][j])
            else:
                row.append(np.nan)
        table_data.append(row)
    
    table = pd.DataFrame(table_data, columns=columns[:n+1])
    
    # Coefficients are the first row (diagonal)
    coefficients = dd[0, :]
    
    # Build polynomial string
    poly_terms = [f"{coefficients[0]:.6f}"]
    for i in range(1, n):
        term = f"{coefficients[i]:+.6f}"
        for j in range(i):
            term += f"(x - {x_points[j]})"
        poly_terms.append(term)
    
    polynomial_str = " ".join(poly_terms)
    
    # Interpolate if target given
    interpolated_value = None
    if x_target is not None:
        result = coefficients[0]
        product = 1
        for i in range(1, n):
            product *= (x_target - x_points[i - 1])
            result += coefficients[i] * product
        interpolated_value = result
    
    return {
        'coefficients': coefficients,
        'polynomial_str': polynomial_str,
        'table': table,
        'interpolated_value': interpolated_value,
        'divided_diff_matrix': dd
    }


def lagrange_interpolation(x_points: np.ndarray, y_points: np.ndarray, x_target: float = None):
    """
    Lagrange Interpolation.
    
    Args:
        x_points: Array of x values (data points)
        y_points: Array of y values (data points)
        x_target: Point to interpolate (optional)
    
    Returns:
        dict with 'basis_polynomials', 'polynomial_str', 'table', 'interpolated_value'
    """
    n = len(x_points)
    
    # Calculate Lagrange basis polynomials at x_target
    table_data = []
    basis_values = []
    
    def L_i(i, x):
        """Calculate L_i(x) - the i-th Lagrange basis polynomial"""
        result = 1.0
        for j in range(n):
            if j != i:
                result *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return result
    
    # Evaluate at target point if given
    interpolated_value = None
    if x_target is not None:
        total = 0.0
        for i in range(n):
            L_i_val = L_i(i, x_target)
            contribution = y_points[i] * L_i_val
            total += contribution
            
            table_data.append({
                'i': i,
                'x_i': x_points[i],
                'y_i': y_points[i],
                f'L_{i}({x_target})': L_i_val,
                f'y_i * L_{i}': contribution
            })
            basis_values.append(L_i_val)
        
        interpolated_value = total
    else:
        # Just build the basis polynomial info
        for i in range(n):
            numerator_parts = []
            denominator_val = 1.0
            for j in range(n):
                if j != i:
                    numerator_parts.append(f"(x - {x_points[j]})")
                    denominator_val *= (x_points[i] - x_points[j])
            
            table_data.append({
                'i': i,
                'x_i': x_points[i],
                'y_i': y_points[i],
                'L_i(x) numerator': " * ".join(numerator_parts),
                'L_i(x) denominator': denominator_val
            })
    
    table = pd.DataFrame(table_data)
    
    # Build polynomial string representation
    poly_parts = []
    for i in range(n):
        term_parts = []
        for j in range(n):
            if j != i:
                term_parts.append(f"(x - {x_points[j]})/({x_points[i]} - {x_points[j]})")
        poly_parts.append(f"{y_points[i]} * " + " * ".join(term_parts))
    
    polynomial_str = " + ".join(poly_parts)
    
    return {
        'basis_values': basis_values if x_target else None,
        'polynomial_str': polynomial_str,
        'table': table,
        'interpolated_value': interpolated_value
    }


def generate_interpolation_curve(x_points: np.ndarray, y_points: np.ndarray, 
                                  method: str = 'newton', num_points: int = 100):
    """
    Generate smooth curve from interpolation for plotting.
    
    Args:
        x_points: Data x values
        y_points: Data y values
        method: 'newton' or 'lagrange'
        num_points: Number of points for smooth curve
    
    Returns:
        tuple of (x_curve, y_curve) for plotting
    """
    x_min, x_max = min(x_points), max(x_points)
    x_curve = np.linspace(x_min, x_max, num_points)
    y_curve = []
    
    for x in x_curve:
        if method == 'newton':
            result = newton_divided_difference(x_points, y_points, x)
        else:
            result = lagrange_interpolation(x_points, y_points, x)
        y_curve.append(result['interpolated_value'])
    
    return x_curve, np.array(y_curve)
