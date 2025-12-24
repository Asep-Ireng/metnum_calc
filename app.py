"""
Numerical Methods Calculator - Main Streamlit Application
A comprehensive calculator implementing 20 numerical methods for solving mathematical problems.

Author: [Student Name]
Course: Metode Numerik
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import method modules
from methods.root_finding import bisection, regula_falsi, newton_raphson, secant
from methods.interpolation import newton_divided_difference, lagrange_interpolation, generate_interpolation_curve
from methods.integration import trapezoidal, simpson_13, simpson_38
from methods.differentiation import forward_difference, backward_difference, central_difference, differentiation_table
from methods.ode import euler_method, heun_method, runge_kutta_4, compare_ode_methods
from methods.linear_systems import gauss_elimination, gauss_jordan, lu_decomposition, jacobi, gauss_seidel
from utils.parser import safe_parse, safe_parse_derivative, safe_parse_ode

# Page configuration
st.set_page_config(
    page_title="Kalkulator Metode Numerik",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .method-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .result-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">🧮 Kalkulator Metode Numerik</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📚 Pilih Kategori")
    category = st.sidebar.radio(
        "Kategori Metode:",
        [
            "🏠 Home",
            "📍 Metode Akar",
            "📈 Interpolasi",
            "∫ Integrasi Numerik",
            "∂ Diferensiasi Numerik",
            "📊 Persamaan Diferensial (ODE)",
            "🔢 Sistem Persamaan Linear"
        ]
    )
    
    if category == "🏠 Home":
        show_home()
    elif category == "📍 Metode Akar":
        show_root_finding()
    elif category == "📈 Interpolasi":
        show_interpolation()
    elif category == "∫ Integrasi Numerik":
        show_integration()
    elif category == "∂ Diferensiasi Numerik":
        show_differentiation()
    elif category == "📊 Persamaan Diferensial (ODE)":
        show_ode()
    elif category == "🔢 Sistem Persamaan Linear":
        show_linear_systems()


def show_home():
    st.markdown("### 👋 Selamat Datang!")
    st.write("""
    Kalkulator ini mengimplementasikan **20 metode numerik** untuk menyelesaikan berbagai permasalahan matematika.
    
    Pilih kategori di sidebar untuk memulai!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Metode Akar (4 metode)")
        st.write("- Bisection\n- Regula Falsi\n- Newton-Raphson\n- Secant")
        
        st.markdown("#### 📈 Interpolasi (2 metode)")
        st.write("- Newton Divided Difference\n- Lagrange")
        
        st.markdown("#### ∫ Integrasi (3 metode)")
        st.write("- Trapezoidal\n- Simpson 1/3\n- Simpson 3/8")
    
    with col2:
        st.markdown("#### ∂ Diferensiasi (3 metode)")
        st.write("- Forward Difference\n- Backward Difference\n- Central Difference")
        
        st.markdown("#### 📊 ODE (3 metode)")
        st.write("- Euler\n- Heun\n- Runge-Kutta 4")
        
        st.markdown("#### 🔢 Sistem Linear (5 metode)")
        st.write("- Gauss Elimination\n- Gauss-Jordan\n- LU Decomposition\n- Jacobi\n- Gauss-Seidel")
    
    st.info("💡 **Tips:** Gunakan format Python untuk fungsi. Contoh: `x**2 - 2`, `sin(x)`, `exp(-x)`")


def show_root_finding():
    st.markdown("## 📍 Metode Pencarian Akar")
    
    method = st.selectbox(
        "Pilih Metode:",
        ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant"]
    )
    
    # Display LaTeX formula for selected method
    st.markdown("### 📐 Formula Metode")
    if method == "Bisection":
        st.latex(r"c = \frac{a + b}{2}")
        st.markdown("""
        **Algoritma:** Bagi interval [a,b] menjadi dua, pilih subinterval yang mengandung akar.
        - Jika f(a)·f(c) < 0, akar di [a,c]
        - Jika f(c)·f(b) < 0, akar di [c,b]
        - **Order konvergensi:** O(1) - Linear
        """)
    elif method == "Regula Falsi":
        st.latex(r"c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}")
        st.markdown("""
        **Algoritma:** Gunakan interpolasi linear untuk estimasi akar.
        - Tarik garis lurus antara (a, f(a)) dan (b, f(b))
        - Titik potong dengan sumbu-x adalah estimasi akar
        - **Order konvergensi:** O(1) - Linear (tapi biasanya lebih cepat dari Bisection)
        """)
    elif method == "Newton-Raphson":
        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
        st.markdown("""
        **Algoritma:** Gunakan garis singgung untuk estimasi akar.
        - Butuh turunan f'(x)
        - Konvergen cepat jika tebakan awal dekat dengan akar
        - **Order konvergensi:** O(2) - Kuadratik
        """)
    else:  # Secant
        st.latex(r"x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}")
        st.markdown("""
        **Algoritma:** Mirip Newton-Raphson tapi tanpa turunan.
        - Aproksimasi turunan dengan selisih terbagi
        - Butuh dua tebakan awal
        - **Order konvergensi:** O(1.618) - Superlinear (Golden Ratio)
        """)
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        func_str = st.text_input("Fungsi f(x):", value="x**3 - x - 2", 
                                  help="Contoh: x**2 - 2, sin(x) - 0.5, exp(-x) - x")
        
        if method in ["Bisection", "Regula Falsi"]:
            c1, c2 = st.columns(2)
            with c1:
                a = st.number_input("Batas bawah (a):", value=1.0)
            with c2:
                b = st.number_input("Batas atas (b):", value=2.0)
            
            # Validate bracketing condition
            try:
                f_test = safe_parse(func_str)
                f_a = f_test(a)
                f_b = f_test(b)
                
                st.markdown("**Syarat Input:**")
                
                # Check a < b
                if a < b:
                    st.markdown("• a < b (✅ OK)")
                else:
                    st.markdown("• a < b (❌ Invalid, a harus lebih kecil dari b)")
                
                # Check bracketing
                if f_a * f_b < 0:
                    st.markdown(f"• f(a)·f(b) < 0 (✅ OK) → f({a})={f_a:.4f}, f({b})={f_b:.4f}")
                else:
                    st.markdown(f"• f(a)·f(b) < 0 (❌ Invalid) → f({a})={f_a:.4f}, f({b})={f_b:.4f}")
                    st.warning("⚠️ Akar tidak terbracket! Ubah interval [a,b]")
            except:
                st.markdown("**Syarat Input:** ⏳ Masukkan fungsi valid untuk validasi")
                
        elif method == "Newton-Raphson":
            x0 = st.number_input("Tebakan awal (x₀):", value=1.5)
            
            # Validate derivative not zero
            try:
                f_test, f_prime_test, _ = safe_parse_derivative(func_str)
                f_prime_x0 = f_prime_test(x0)
                
                st.markdown("**Syarat Input:**")
                if abs(f_prime_x0) > 1e-10:
                    st.markdown(f"• f'(x₀) ≠ 0 (✅ OK) → f'({x0})={f_prime_x0:.6f}")
                else:
                    st.markdown(f"• f'(x₀) ≠ 0 (❌ Invalid) → f'({x0})≈0")
                    st.warning("⚠️ Turunan mendekati nol! Ubah tebakan awal")
            except:
                st.markdown("**Syarat Input:** ⏳ Masukkan fungsi valid untuk validasi")
                
        else:  # Secant
            c1, c2 = st.columns(2)
            with c1:
                x0 = st.number_input("Tebakan awal pertama (x₀):", value=1.0)
            with c2:
                x1 = st.number_input("Tebakan awal kedua (x₁):", value=2.0)
            
            # Validate x0 != x1
            try:
                f_test = safe_parse(func_str)
                f_x0 = f_test(x0)
                f_x1 = f_test(x1)
                
                st.markdown("**Syarat Input:**")
                
                if x0 != x1:
                    st.markdown("• x₀ ≠ x₁ (✅ OK)")
                else:
                    st.markdown("• x₀ ≠ x₁ (❌ Invalid)")
                
                if abs(f_x1 - f_x0) > 1e-10:
                    st.markdown(f"• f(x₀) ≠ f(x₁) (✅ OK) → f({x0})={f_x0:.4f}, f({x1})={f_x1:.4f}")
                else:
                    st.markdown(f"• f(x₀) ≠ f(x₁) (❌ Invalid) → nilai sama")
            except:
                st.markdown("**Syarat Input:** ⏳ Masukkan fungsi valid untuk validasi")
        
        c1, c2 = st.columns(2)
        with c1:
            tol = st.number_input("Toleransi:", value=1e-6, format="%.2e")
        with c2:
            max_iter = st.number_input("Maksimum iterasi:", value=100, min_value=1)
        
        exact_str = st.text_input("Nilai akar eksak (opsional, untuk error):", value="",
                                   help="Masukkan nilai akar eksak jika diketahui untuk menghitung error")
    
    if st.button("🔍 Hitung Akar", type="primary"):
        try:
            if method == "Newton-Raphson":
                f, f_prime, deriv_str = safe_parse_derivative(func_str)
                st.info(f"Turunan f'(x) = {deriv_str}")
                result = newton_raphson(f, f_prime, x0, tol, max_iter)
            else:
                f = safe_parse(func_str)
                if method == "Bisection":
                    result = bisection(f, a, b, tol, max_iter)
                elif method == "Regula Falsi":
                    result = regula_falsi(f, a, b, tol, max_iter)
                else:  # Secant
                    result = secant(f, x0, x1, tol, max_iter)
            
            if result.get('error'):
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success(f"✅ **Akar ditemukan: x = {result['root']:.10f}**")
                st.write(f"Jumlah iterasi: {result['iterations']}")
                st.write(f"Error aproksimasi: {result.get('final_error', 'N/A'):.2e}%")
                
                # Calculate true error if exact value provided
                if exact_str:
                    try:
                        exact = float(exact_str)
                        true_error = abs((result['root'] - exact) / exact) * 100 if exact != 0 else abs(result['root'] - exact)
                        st.write(f"Error terhadap nilai eksak: {true_error:.6f}%")
                    except:
                        pass
                
                # Show iteration table
                st.markdown("### 📋 Tabel Iterasi")
                st.dataframe(result['table'], use_container_width=True)
                
                # Function Plot - NEW!
                st.markdown("### 📊 Grafik Fungsi f(x)")
                
                # Determine plot range
                if method in ["Bisection", "Regula Falsi"]:
                    x_min, x_max = min(a, result['root']) - 0.5, max(b, result['root']) + 0.5
                else:
                    x_min, x_max = result['root'] - 2, result['root'] + 2
                
                x_plot = np.linspace(x_min, x_max, 200)
                y_plot = [f(x) for x in x_plot]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {func_str}')
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.scatter([result['root']], [0], color='red', s=150, zorder=5, 
                          label=f'Akar x = {result["root"]:.6f}', marker='o')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.set_title(f'Grafik Fungsi: f(x) = {func_str}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Convergence Plot
                st.markdown("### 📈 Grafik Konvergensi")
                if method in ["Bisection", "Regula Falsi"]:
                    plot_col = 'c'
                elif method == "Newton-Raphson":
                    plot_col = 'x'
                else:
                    plot_col = 'x_{n+1}'
                
                if plot_col in result['table'].columns:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                    
                    # Left: x values convergence
                    ax1.plot(result['table']['Iterasi'], result['table'][plot_col], 'bo-', markersize=8)
                    ax1.axhline(y=result['root'], color='r', linestyle='--', label=f'Akar = {result["root"]:.6f}')
                    ax1.set_xlabel('Iterasi')
                    ax1.set_ylabel('Nilai x')
                    ax1.set_title(f'Konvergensi Nilai x')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Right: Error convergence (log scale)
                    if 'Error (%)' in result['table'].columns:
                        errors = result['table']['Error (%)'].tolist()
                        valid_errors = [(i, e) for i, e in enumerate(errors) if isinstance(e, (int, float)) and e > 0]
                        if valid_errors:
                            iters, errs = zip(*valid_errors)
                            ax2.semilogy(iters, errs, 'ro-', markersize=8)
                            ax2.set_xlabel('Iterasi')
                            ax2.set_ylabel('Error (%) - Log Scale')
                            ax2.set_title('Laju Konvergensi Error')
                            ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Convergence Rate Analysis - NEW!
                st.markdown("### 📉 Analisis Laju Konvergensi")
                if 'Error (%)' in result['table'].columns:
                    errors = [e for e in result['table']['Error (%)'].tolist() if isinstance(e, (int, float)) and e > 0]
                    if len(errors) >= 3:
                        # Calculate convergence rate
                        ratios = []
                        for i in range(1, len(errors)):
                            if errors[i-1] > 0:
                                ratios.append(errors[i] / errors[i-1])
                        
                        if ratios:
                            avg_ratio = np.mean(ratios)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Error Awal", f"{errors[0]:.4f}%")
                            with col2:
                                st.metric("Error Akhir", f"{errors[-1]:.2e}%")
                            with col3:
                                st.metric("Rata-rata Rasio Konvergensi", f"{avg_ratio:.4f}")
                            
                            if avg_ratio < 0.1:
                                st.success("🚀 **Konvergensi sangat cepat** (kuadratik atau lebih tinggi)")
                            elif avg_ratio < 0.5:
                                st.info("⚡ **Konvergensi cepat** (superlinear)")
                            else:
                                st.warning("📊 **Konvergensi linear**")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_interpolation():
    st.markdown("## 📈 Metode Interpolasi")
    
    method = st.selectbox("Pilih Metode:", ["Newton Divided Difference", "Lagrange"])
    
    st.markdown("---")
    
    # Input data points
    st.markdown("### 📊 Input Data Points")
    
    input_method = st.radio("Metode input:", ["Manual", "Tabel"], horizontal=True)
    
    if input_method == "Manual":
        x_str = st.text_input("Nilai x (pisahkan dengan koma):", value="1, 2, 3, 4, 5")
        y_str = st.text_input("Nilai y (pisahkan dengan koma):", value="1, 4, 9, 16, 25")
        
        try:
            x_points = np.array([float(x.strip()) for x in x_str.split(',')])
            y_points = np.array([float(y.strip()) for y in y_str.split(',')])
        except:
            st.error("Format input salah!")
            return
    else:
        n_points = st.number_input("Jumlah titik data:", min_value=2, value=5)
        
        cols = st.columns(n_points)
        x_points = []
        y_points = []
        for i, col in enumerate(cols):
            with col:
                x_points.append(st.number_input(f"x{i+1}", value=float(i+1), key=f"x_{i}"))
                y_points.append(st.number_input(f"y{i+1}", value=float((i+1)**2), key=f"y_{i}"))
        x_points = np.array(x_points)
        y_points = np.array(y_points)
    
    x_target = st.number_input("Nilai x yang akan diinterpolasi:", value=2.5)
    exact_str = st.text_input("Nilai eksak f(x_target) (opsional, untuk error):", value="",
                               help="Masukkan nilai eksak hasil interpolasi jika diketahui")
    
    # Validation
    st.markdown("**Syarat Input:**")
    
    # Check data points count
    n = len(x_points)
    if n >= 2:
        st.markdown(f"• Jumlah titik ≥ 2 (✅ OK) → {n} titik")
    else:
        st.markdown("• Jumlah titik ≥ 2 (❌ Invalid)")
    
    # Check unique x values
    if len(set(x_points)) == len(x_points):
        st.markdown("• Nilai x unik (✅ OK)")
    else:
        st.markdown("• Nilai x unik (❌ Invalid, ada nilai x duplikat)")
    
    # Check length match
    if len(x_points) == len(y_points):
        st.markdown("• Jumlah x = Jumlah y (✅ OK)")
    else:
        st.markdown("• Jumlah x = Jumlah y (❌ Invalid)")
    
    # Check if target is within range (interpolation vs extrapolation)
    if len(x_points) > 0:
        x_min, x_max = min(x_points), max(x_points)
        if x_min <= x_target <= x_max:
            st.markdown(f"• x_target dalam range (✅ Interpolasi) → [{x_min}, {x_max}]")
        else:
            st.markdown(f"• x_target diluar range (⚠️ Ekstrapolasi) → target={x_target}, range=[{x_min}, {x_max}]")
    
    if st.button("📈 Hitung Interpolasi", type="primary"):
        try:
            if method == "Newton Divided Difference":
                result = newton_divided_difference(x_points, y_points, x_target)
            else:
                result = lagrange_interpolation(x_points, y_points, x_target)
            
            st.success(f"✅ **Hasil interpolasi f({x_target}) = {result['interpolated_value']:.10f}**")
            
            # Calculate error if exact value provided
            if exact_str:
                try:
                    exact = float(exact_str)
                    error = abs((result['interpolated_value'] - exact) / exact) * 100 if exact != 0 else abs(result['interpolated_value'] - exact)
                    st.write(f"Error terhadap nilai eksak: {error:.6f}%")
                except:
                    pass
            
            # Show table
            st.markdown("### 📋 Tabel Perhitungan")
            st.dataframe(result['table'], use_container_width=True)
            
            # Plot
            st.markdown("### 📈 Grafik Interpolasi")
            x_curve, y_curve = generate_interpolation_curve(x_points, y_points, 
                                                            'newton' if method == "Newton Divided Difference" else 'lagrange')
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_curve, y_curve, 'b-', label='Kurva Interpolasi', linewidth=2)
            ax.scatter(x_points, y_points, color='red', s=100, zorder=5, label='Data Points')
            ax.scatter([x_target], [result['interpolated_value']], color='green', s=150, 
                      marker='X', zorder=6, label=f'Interpolasi ({x_target}, {result["interpolated_value"]:.4f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Interpolasi {method}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_integration():
    st.markdown("## ∫ Integrasi Numerik")
    
    method = st.selectbox("Pilih Metode:", ["Trapezoidal", "Simpson 1/3", "Simpson 3/8", "Bandingkan Semua"])
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        func_str = st.text_input("Fungsi f(x):", value="x**2", 
                                  help="Contoh: x**2, sin(x), exp(-x)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            a = st.number_input("Batas bawah (a):", value=0.0)
        with c2:
            b = st.number_input("Batas atas (b):", value=1.0)
        with c3:
            n = st.number_input("Jumlah sub-interval (n):", value=10, min_value=1)
        
        # Syarat N validation display
        st.markdown("**Syarat N:**")
        
        # Trapesium - always valid
        st.markdown("• Trapesium: Bebas ✅")
        
        # Simpson 1/3 - must be even
        if n % 2 == 0:
            st.markdown("• Simpson 1/3: Genap (✅ OK)")
        else:
            st.markdown("• Simpson 1/3: Genap (❌ Invalid, n harus genap)")
        
        # Simpson 3/8 - must be multiple of 3
        if n % 3 == 0:
            st.markdown("• Simpson 3/8: Kelipatan 3 (✅ OK)")
        else:
            st.markdown("• Simpson 3/8: Kelipatan 3 (❌ Invalid, n harus kelipatan 3)")
        
        # Additional bounds check
        if a >= b:
            st.error("❌ Batas bawah (a) harus lebih kecil dari batas atas (b)!")
        
        exact_str = st.text_input("Nilai eksak (opsional, untuk error):", value="", 
                                   help="Masukkan nilai eksak integral jika diketahui")
    
    if st.button("∫ Hitung Integral", type="primary"):
        try:
            f = safe_parse(func_str)
            exact = float(exact_str) if exact_str else None
            
            if method == "Bandingkan Semua":
                st.markdown("### 📊 Perbandingan Metode")
                
                results = []
                
                # Trapezoidal - always valid
                trap = trapezoidal(f, a, b, n)
                results.append({'Metode': 'Trapezoidal', 'n': n, 'Hasil': trap['integral']})
                
                # Simpson 1/3 - only if n is even
                if n % 2 == 0:
                    simp13 = simpson_13(f, a, b, n)
                    if simp13['integral'] is not None:
                        results.append({'Metode': 'Simpson 1/3', 'n': n, 'Hasil': simp13['integral']})
                
                # Simpson 3/8 - only if n is multiple of 3
                if n % 3 == 0:
                    simp38 = simpson_38(f, a, b, n)
                    if simp38['integral'] is not None:
                        results.append({'Metode': 'Simpson 3/8', 'n': n, 'Hasil': simp38['integral']})
                
                if exact:
                    for r in results:
                        r['Error (%)'] = abs((r['Hasil'] - exact) / exact) * 100
                    results.append({'Metode': 'Eksak', 'n': '-', 'Hasil': exact, 'Error (%)': 0})
                
                # Show info about excluded methods
                excluded = []
                if n % 2 != 0:
                    excluded.append("Simpson 1/3 (n harus genap)")
                if n % 3 != 0:
                    excluded.append("Simpson 3/8 (n harus kelipatan 3)")
                
                if excluded:
                    st.info(f"ℹ️ Metode tidak ditampilkan karena n={n} invalid: {', '.join(excluded)}")
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                if method == "Trapezoidal":
                    result = trapezoidal(f, a, b, n)
                elif method == "Simpson 1/3":
                    if n % 2 != 0:
                        st.warning(f"⚠️ n harus genap untuk Simpson 1/3. Menggunakan n = {n+1}")
                        n = n + 1
                    result = simpson_13(f, a, b, n)
                else:  # Simpson 3/8
                    if n % 3 != 0:
                        n_new = (n // 3 + 1) * 3
                        st.warning(f"⚠️ n harus kelipatan 3 untuk Simpson 3/8. Menggunakan n = {n_new}")
                        n = n_new
                    result = simpson_38(f, a, b, n)
                
                if result.get('error'):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.success(f"✅ **Hasil integral: {result['integral']:.10f}**")
                    st.write(f"Step size h = {result['h']:.6f}")
                    
                    if exact:
                        error = abs((result['integral'] - exact) / exact) * 100
                        st.write(f"Error relatif: {error:.6f}%")
                    
                    st.markdown("### 📋 Tabel Perhitungan")
                    st.dataframe(result['table'], use_container_width=True)
                    
                    # Visualization
                    st.markdown("### 📈 Visualisasi Area")
                    x_plot = np.linspace(a, b, 100)
                    y_plot = [f(x) for x in x_plot]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
                    ax.fill_between(x_plot, 0, y_plot, alpha=0.3)
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)')
                    ax.set_title(f'Integrasi {method}: ∫f(x)dx = {result["integral"]:.6f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_differentiation():
    st.markdown("## ∂ Diferensiasi Numerik")
    
    method = st.selectbox("Pilih Metode:", 
                          ["Forward Difference", "Backward Difference", "Central Difference", "Bandingkan Semua"])
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        func_str = st.text_input("Fungsi f(x):", value="sin(x)", 
                                  help="Contoh: x**2, sin(x), exp(-x)")
        
        c1, c2 = st.columns(2)
        with c1:
            x = st.number_input("Titik evaluasi x:", value=1.0)
        with c2:
            h = st.number_input("Step size h:", value=0.1, format="%.4f")
        
        # Validation
        st.markdown("**Syarat Input:**")
        
        if h > 0:
            st.markdown(f"• h > 0 (✅ OK) → h = {h}")
        else:
            st.markdown("• h > 0 (❌ Invalid, step size harus positif)")
        
        if h <= 1:
            st.markdown("• h ≤ 1 (✅ Recommended untuk akurasi baik)")
        else:
            st.markdown("• h > 1 (⚠️ Warning: h besar dapat mengurangi akurasi)")
        
        exact_str = st.text_input("Nilai turunan eksak (opsional):", value="",
                                   help="Masukkan f'(x) eksak jika diketahui")
    
    if st.button("∂ Hitung Turunan", type="primary"):
        try:
            f = safe_parse(func_str)
            exact = float(exact_str) if exact_str else None
            
            if method == "Bandingkan Semua":
                st.markdown("### 📊 Perbandingan Metode")
                
                table = differentiation_table(f, x, [1, 0.5, 0.1, 0.05, 0.01], exact)
                st.dataframe(table, use_container_width=True)
                
                if exact:
                    st.info(f"Nilai eksak f'({x}) = {exact}")
            else:
                if method == "Forward Difference":
                    result = forward_difference(f, x, h)
                elif method == "Backward Difference":
                    result = backward_difference(f, x, h)
                else:  # Central
                    result = central_difference(f, x, h)
                
                st.success(f"✅ **Hasil turunan f'({x}) ≈ {result['derivative']:.10f}**")
                st.write(f"Formula: {result['formula']}")
                st.write(f"Order: {result['order']}")
                
                if exact:
                    error = abs((result['derivative'] - exact) / exact) * 100
                    st.write(f"Error relatif: {error:.6f}%")
                
                # Show details
                st.markdown("### 📋 Detail Perhitungan")
                details = {k: v for k, v in result.items() if k not in ['formula', 'order']}
                st.json(details)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_ode():
    st.markdown("## 📊 Penyelesaian Persamaan Diferensial (ODE)")
    st.write("Menyelesaikan dy/dx = f(x, y) dengan kondisi awal y(x₀) = y₀")
    
    method = st.selectbox("Pilih Metode:", 
                          ["Euler", "Heun (Improved Euler)", "Runge-Kutta 4", "Bandingkan Semua"])
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        func_str = st.text_input("Fungsi f(x, y) = dy/dx:", value="y", 
                                  help="Contoh: y (untuk dy/dx = y), x + y, -2*x*y")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            x0 = st.number_input("x₀ (nilai awal x):", value=0.0)
            y0 = st.number_input("y₀ (nilai awal y):", value=1.0)
        with c2:
            x_end = st.number_input("x akhir:", value=2.0)
        with c3:
            h = st.number_input("Step size h:", value=0.2, format="%.4f")
        
        # Validation
        st.markdown("**Syarat Input:**")
        
        if x_end > x0:
            st.markdown(f"• x_end > x₀ (✅ OK) → interval [{x0}, {x_end}]")
        else:
            st.markdown("• x_end > x₀ (❌ Invalid, x akhir harus lebih besar)")
        
        if h > 0:
            st.markdown(f"• h > 0 (✅ OK) → h = {h}")
        else:
            st.markdown("• h > 0 (❌ Invalid, step size harus positif)")
        
        if h > 0 and x_end > x0:
            n_steps = int((x_end - x0) / h)
            st.markdown(f"• Jumlah langkah: {n_steps} steps")
            
            if n_steps > 1000:
                st.warning("⚠️ Banyak langkah! Perhitungan mungkin lambat")
            elif n_steps < 2:
                st.warning("⚠️ Step size terlalu besar, kurangi h")
        
        exact_str = st.text_input("Solusi eksak y(x) (opsional):", value="exp(x)",
                                   help="Untuk perhitungan error. Contoh: exp(x)")
    
    if st.button("📊 Selesaikan ODE", type="primary"):
        try:
            f = safe_parse_ode(func_str)
            exact_func = safe_parse(exact_str) if exact_str else None
            
            if method == "Bandingkan Semua":
                comparison = compare_ode_methods(f, x0, y0, h, x_end, exact_func)
                
                st.markdown("### 📊 Perbandingan Metode")
                st.dataframe(comparison['comparison_table'], use_container_width=True)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(comparison['euler']['x_values'], comparison['euler']['y_values'], 
                       'b-o', label='Euler', markersize=4)
                ax.plot(comparison['heun']['x_values'], comparison['heun']['y_values'], 
                       'g-s', label='Heun', markersize=4)
                ax.plot(comparison['rk4']['x_values'], comparison['rk4']['y_values'], 
                       'r-^', label='RK4', markersize=4)
                
                if exact_func:
                    x_exact = np.linspace(x0, x_end, 100)
                    y_exact = [exact_func(x) for x in x_exact]
                    ax.plot(x_exact, y_exact, 'k--', label='Exact', linewidth=2)
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Perbandingan Metode ODE')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                if method == "Euler":
                    result = euler_method(f, x0, y0, h, x_end)
                elif method == "Heun (Improved Euler)":
                    result = heun_method(f, x0, y0, h, x_end)
                else:  # RK4
                    result = runge_kutta_4(f, x0, y0, h, x_end)
                
                st.success(f"✅ **Solusi dihitung untuk x ∈ [{x0}, {x_end}]**")
                st.write(f"Nilai akhir y({x_end}) = {result['y_values'][-1]:.10f}")
                
                st.markdown("### 📋 Tabel Iterasi")
                st.dataframe(result['table'], use_container_width=True)
                
                # Plot
                st.markdown("### 📈 Grafik Solusi")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(result['x_values'], result['y_values'], 'b-o', label=method, markersize=6)
                
                if exact_func:
                    x_exact = np.linspace(x0, x_end, 100)
                    y_exact = [exact_func(x) for x in x_exact]
                    ax.plot(x_exact, y_exact, 'r--', label='Exact', linewidth=2)
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Solusi ODE dengan Metode {method}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_linear_systems():
    st.markdown("## 🔢 Sistem Persamaan Linear")
    st.write("Menyelesaikan Ax = b")
    
    method = st.selectbox("Pilih Metode:", 
                          ["Gauss Elimination", "Gauss-Jordan", "LU Decomposition", 
                           "Jacobi (Iteratif)", "Gauss-Seidel (Iteratif)"])
    
    st.markdown("---")
    
    # Input matrix size
    n = st.number_input("Ukuran matriks (n x n):", min_value=2, max_value=10, value=3)
    
    st.markdown("### 📊 Input Matriks A dan Vektor b")
    
    # Input matrix A
    st.write("Matriks koefisien A:")
    A = np.zeros((n, n))
    
    # Default example matrix
    default_A = np.array([
        [4, -1, 0],
        [-1, 4, -1],
        [0, -1, 4]
    ])
    default_b = np.array([15, 10, 15])
    
    cols = st.columns(n)
    for j in range(n):
        with cols[j]:
            for i in range(n):
                default_val = float(default_A[i, j]) if i < 3 and j < 3 else 0.0
                A[i, j] = st.number_input(f"A[{i+1},{j+1}]", value=default_val, key=f"A_{i}_{j}")
    
    st.write("Vektor konstanta b:")
    b = np.zeros(n)
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            default_val = float(default_b[i]) if i < 3 else 0.0
            b[i] = st.number_input(f"b[{i+1}]", value=default_val, key=f"b_{i}")
    
    # Validation
    st.markdown("**Syarat Input:**")
    
    # Check determinant (non-singular)
    try:
        det = np.linalg.det(A)
        if abs(det) > 1e-10:
            st.markdown(f"• det(A) ≠ 0 (✅ OK) → det = {det:.6f}")
        else:
            st.markdown(f"• det(A) ≠ 0 (❌ Invalid) → det ≈ 0, matriks singular")
            st.error("❌ Matriks singular! Tidak ada solusi unik")
    except:
        st.markdown("• det(A): ⏳ Checking...")
    
    # Check diagonal dominance for iterative methods
    if method in ["Jacobi (Iteratif)", "Gauss-Seidel (Iteratif)"]:
        diag_dominant = True
        for i in range(n):
            diag = abs(A[i, i])
            off_diag_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
            if diag <= off_diag_sum:
                diag_dominant = False
                break
        
        if diag_dominant:
            st.markdown("• Diagonal dominan (✅ OK) → konvergensi dijamin")
        else:
            st.markdown("• Diagonal dominan (⚠️ Warning) → konvergensi tidak dijamin")
            st.warning("⚠️ Matriks tidak diagonal dominan. Metode iteratif mungkin tidak konvergen.")
    
    # Extra parameters for iterative methods
    if method in ["Jacobi (Iteratif)", "Gauss-Seidel (Iteratif)"]:
        c1, c2 = st.columns(2)
        with c1:
            tol = st.number_input("Toleransi:", value=1e-6, format="%.2e")
        with c2:
            max_iter = st.number_input("Maksimum iterasi:", value=100, min_value=1)
    
    if st.button("🔢 Selesaikan SPL", type="primary"):
        try:
            A = np.array(A)
            b = np.array(b)
            
            if method == "Gauss Elimination":
                result = gauss_elimination(A, b)
            elif method == "Gauss-Jordan":
                result = gauss_jordan(A, b)
            elif method == "LU Decomposition":
                result = lu_decomposition(A, b)
            elif method == "Jacobi (Iteratif)":
                result = jacobi(A, b, tol=tol, max_iter=max_iter)
            else:  # Gauss-Seidel
                result = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
            
            if result.get('error'):
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ **Solusi ditemukan:**")
                
                # Display solution
                sol_df = pd.DataFrame({
                    'Variabel': [f'x{i+1}' for i in range(n)],
                    'Nilai': result['solution']
                })
                st.dataframe(sol_df, use_container_width=True)
                
                # Method-specific outputs
                if method == "LU Decomposition":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Matriks L")
                        st.dataframe(pd.DataFrame(result['L']).round(6))
                    with col2:
                        st.markdown("#### Matriks U")
                        st.dataframe(pd.DataFrame(result['U']).round(6))
                
                elif method in ["Gauss Elimination", "Gauss-Jordan"]:
                    st.markdown("### 📋 Langkah-langkah Eliminasi")
                    for step in result.get('steps', []):
                        st.write(f"**{step['step']}**")
                        st.dataframe(pd.DataFrame(step['matrix']).round(6))
                
                elif method in ["Jacobi (Iteratif)", "Gauss-Seidel (Iteratif)"]:
                    st.write(f"Jumlah iterasi: {result['iterations']}")
                    st.write(f"Error akhir: {result.get('final_error', 'N/A'):.2e}")
                    
                    if not result.get('diagonal_dominant'):
                        st.warning("⚠️ Matriks tidak diagonal dominan. Konvergensi tidak dijamin.")
                    
                    st.markdown("### 📋 Tabel Iterasi")
                    st.dataframe(result['table'], use_container_width=True)
                    
                    # Convergence plot
                    if 'Error' in result['table'].columns:
                        errors = [e for e in result['table']['Error'].tolist()[1:] if isinstance(e, (int, float))]
                        if errors:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.semilogy(range(1, len(errors)+1), errors, 'bo-')
                            ax.set_xlabel('Iterasi')
                            ax.set_ylabel('Error (log scale)')
                            ax.set_title(f'Konvergensi {method}')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
