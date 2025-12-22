"""
Function Parser Utility
Safely parses user input string to callable Python functions using sympy.
"""
import numpy as np
from sympy import symbols, sympify, lambdify, diff, exp, sin, cos, tan, log, sqrt, pi, E
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


def safe_parse(expr_str: str, variable: str = 'x'):
    """
    Parse a mathematical expression string into a callable function.
    
    Args:
        expr_str: Mathematical expression as string (e.g., "x**2 - 2", "sin(x)")
        variable: Variable name used in the expression (default: 'x')
    
    Returns:
        A callable function that takes a numpy array and returns evaluated values
    
    Examples:
        >>> f = safe_parse("x**2 - 2")
        >>> f(2)  # Returns 2
        >>> f = safe_parse("sin(x) + cos(x)")
        >>> f(0)  # Returns 1
    """
    try:
        # Define the variable symbol
        var = symbols(variable)
        
        # Parse with transformations to handle implicit multiplication
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        # Replace common function names for compatibility
        expr_str = expr_str.replace('^', '**')  # Support ^ for exponents
        expr_str = expr_str.replace('e^', 'exp')  # Support e^ notation
        
        # Parse the expression
        expr = parse_expr(expr_str, transformations=transformations)
        
        # Create a numpy-compatible function
        func = lambdify(var, expr, modules=['numpy'])
        
        return func
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expr_str}': {str(e)}")


def safe_parse_derivative(expr_str: str, variable: str = 'x'):
    """
    Parse expression and return both function and its derivative.
    
    Args:
        expr_str: Mathematical expression as string
        variable: Variable name
    
    Returns:
        Tuple of (function, derivative_function)
    """
    try:
        var = symbols(variable)
        
        # Parse expression
        expr_str = expr_str.replace('^', '**')
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(expr_str, transformations=transformations)
        
        # Compute derivative
        derivative_expr = diff(expr, var)
        
        # Create numpy-compatible functions
        func = lambdify(var, expr, modules=['numpy'])
        derivative_func = lambdify(var, derivative_expr, modules=['numpy'])
        
        return func, derivative_func, str(derivative_expr)
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expr_str}': {str(e)}")


def safe_parse_ode(expr_str: str, x_var: str = 'x', y_var: str = 'y'):
    """
    Parse an ODE expression f(x, y) for dy/dx = f(x, y).
    
    Args:
        expr_str: Expression in terms of x and y
        x_var: Independent variable name
        y_var: Dependent variable name
    
    Returns:
        Callable function f(x, y)
    """
    try:
        x = symbols(x_var)
        y = symbols(y_var)
        
        expr_str = expr_str.replace('^', '**')
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(expr_str, transformations=transformations)
        
        func = lambdify((x, y), expr, modules=['numpy'])
        
        return func
    except Exception as e:
        raise ValueError(f"Error parsing ODE expression '{expr_str}': {str(e)}")


def get_expression_latex(expr_str: str) -> str:
    """
    Convert expression string to LaTeX format for display.
    """
    try:
        from sympy import latex
        expr_str = expr_str.replace('^', '**')
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(expr_str, transformations=transformations)
        return latex(expr)
    except:
        return expr_str
