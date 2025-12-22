"""
Linear Systems Solvers Module
Implements: Gauss Elimination, Gauss-Jordan, LU Decomposition, Jacobi, Gauss-Seidel
"""
import numpy as np
import pandas as pd


def gauss_elimination(A: np.ndarray, b: np.ndarray):
    """
    Gauss Elimination with back substitution.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Constant vector (n)
    
    Returns:
        dict with 'solution', 'steps', 'augmented_matrix'
    """
    n = len(b)
    # Create augmented matrix
    Aug = np.column_stack([A.astype(float), b.astype(float)])
    
    steps = []
    steps.append({
        'step': 'Matriks Augmented Awal',
        'matrix': Aug.copy()
    })
    
    # Forward elimination
    for k in range(n - 1):
        # Find pivot
        max_idx = np.argmax(np.abs(Aug[k:, k])) + k
        if Aug[max_idx, k] == 0:
            return {
                'solution': None,
                'steps': steps,
                'error': 'Matriks singular, tidak ada solusi unik'
            }
        
        # Swap rows if needed
        if max_idx != k:
            Aug[[k, max_idx]] = Aug[[max_idx, k]]
            steps.append({
                'step': f'Tukar baris {k+1} dengan baris {max_idx+1}',
                'matrix': Aug.copy()
            })
        
        # Eliminate
        for i in range(k + 1, n):
            if Aug[k, k] != 0:
                factor = Aug[i, k] / Aug[k, k]
                Aug[i, k:] = Aug[i, k:] - factor * Aug[k, k:]
        
        steps.append({
            'step': f'Eliminasi kolom {k+1}',
            'matrix': Aug.copy()
        })
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if Aug[i, i] == 0:
            return {
                'solution': None,
                'steps': steps,
                'error': 'Pembagi nol dalam back substitution'
            }
        x[i] = (Aug[i, -1] - np.dot(Aug[i, i+1:n], x[i+1:n])) / Aug[i, i]
    
    return {
        'solution': x,
        'steps': steps,
        'final_matrix': Aug
    }


def gauss_jordan(A: np.ndarray, b: np.ndarray):
    """
    Gauss-Jordan Elimination (reduced row echelon form).
    
    Args:
        A: Coefficient matrix (n x n)
        b: Constant vector (n)
    
    Returns:
        dict with 'solution', 'steps', 'augmented_matrix'
    """
    n = len(b)
    Aug = np.column_stack([A.astype(float), b.astype(float)])
    
    steps = []
    steps.append({
        'step': 'Matriks Augmented Awal',
        'matrix': Aug.copy()
    })
    
    for k in range(n):
        # Find pivot
        max_idx = np.argmax(np.abs(Aug[k:, k])) + k
        if Aug[max_idx, k] == 0:
            return {
                'solution': None,
                'steps': steps,
                'error': 'Matriks singular'
            }
        
        # Swap rows if needed
        if max_idx != k:
            Aug[[k, max_idx]] = Aug[[max_idx, k]]
        
        # Scale pivot row
        Aug[k] = Aug[k] / Aug[k, k]
        
        # Eliminate all other rows
        for i in range(n):
            if i != k:
                Aug[i] = Aug[i] - Aug[i, k] * Aug[k]
        
        steps.append({
            'step': f'Pivot kolom {k+1}, eliminasi semua baris lain',
            'matrix': Aug.copy()
        })
    
    # Solution is last column
    x = Aug[:, -1]
    
    return {
        'solution': x,
        'steps': steps,
        'final_matrix': Aug
    }


def lu_decomposition(A: np.ndarray, b: np.ndarray):
    """
    LU Decomposition: A = LU, then solve Ly = b, Ux = y.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Constant vector (n)
    
    Returns:
        dict with 'solution', 'L', 'U', 'y'
    """
    n = len(b)
    L = np.eye(n)
    U = A.astype(float).copy()
    
    # Decomposition
    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                return {
                    'solution': None,
                    'error': 'Pembagi nol dalam dekomposisi LU'
                }
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] = U[i, k:] - factor * U[k, k:]
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Backward substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            return {
                'solution': None,
                'error': 'Pembagi nol dalam back substitution'
            }
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return {
        'solution': x,
        'L': L,
        'U': U,
        'y': y
    }


def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None, 
           tol: float = 1e-6, max_iter: int = 100):
    """
    Jacobi Iterative Method for solving linear systems.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Constant vector (n)
        x0: Initial guess (optional)
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        dict with 'solution', 'iterations', 'table', 'converged'
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    
    # Check diagonal dominance (warning only)
    diag_dominant = True
    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            diag_dominant = False
            break
    
    # Initial guess
    x = np.zeros(n) if x0 is None else x0.astype(float)
    
    iterations = []
    iterations.append({
        'Iterasi': 0,
        **{f'x{i+1}': x[i] for i in range(n)},
        'Error': '-'
    })
    
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        x_new = np.zeros(n)
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x_old[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            if A[i, i] == 0:
                return {
                    'solution': None,
                    'error': f'Elemen diagonal A[{i},{i}] = 0'
                }
            x_new[i] = (b[i] - sigma) / A[i, i]
        
        x = x_new
        
        # Calculate error
        error = np.max(np.abs(x - x_old))
        
        iterations.append({
            'Iterasi': k,
            **{f'x{i+1}': x[i] for i in range(n)},
            'Error': error
        })
        
        if error < tol:
            break
    
    return {
        'solution': x,
        'iterations': k,
        'table': pd.DataFrame(iterations),
        'converged': error < tol,
        'diagonal_dominant': diag_dominant,
        'final_error': error
    }


def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
                  tol: float = 1e-6, max_iter: int = 100):
    """
    Gauss-Seidel Iterative Method for solving linear systems.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Constant vector (n)
        x0: Initial guess (optional)
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        dict with 'solution', 'iterations', 'table', 'converged'
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    
    # Check diagonal dominance
    diag_dominant = True
    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            diag_dominant = False
            break
    
    # Initial guess
    x = np.zeros(n) if x0 is None else x0.astype(float)
    
    iterations = []
    iterations.append({
        'Iterasi': 0,
        **{f'x{i+1}': x[i] for i in range(n)},
        'Error': '-'
    })
    
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        
        for i in range(n):
            # Use updated values immediately (main difference from Jacobi)
            sigma1 = np.dot(A[i, :i], x[:i])
            sigma2 = np.dot(A[i, i+1:], x_old[i+1:])
            if A[i, i] == 0:
                return {
                    'solution': None,
                    'error': f'Elemen diagonal A[{i},{i}] = 0'
                }
            x[i] = (b[i] - sigma1 - sigma2) / A[i, i]
        
        # Calculate error
        error = np.max(np.abs(x - x_old))
        
        iterations.append({
            'Iterasi': k,
            **{f'x{i+1}': x[i] for i in range(n)},
            'Error': error
        })
        
        if error < tol:
            break
    
    return {
        'solution': x,
        'iterations': k,
        'table': pd.DataFrame(iterations),
        'converged': error < tol,
        'diagonal_dominant': diag_dominant,
        'final_error': error
    }


def format_matrix(matrix: np.ndarray, name: str = "Matrix"):
    """Helper function to format matrix for display."""
    df = pd.DataFrame(matrix)
    df.columns = [f'Col {i+1}' for i in range(matrix.shape[1])]
    df.index = [f'Row {i+1}' for i in range(matrix.shape[0])]
    return df
