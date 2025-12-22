# Import library numpy untuk perhitungan numerik yang efisien
import numpy as np


def gauss_seidel(A, b, x0=None, tolerance=1e-6, max_iterations=100):
    """
    Menyelesaikan sistem persamaan linear Ax = b menggunakan metode Gauss-Seidel.

    Args:
        A (np.array): Matriks koefisien (harus persegi).
        b (np.array): Vektor konstanta (sisi kanan).
        x0 (np.array, optional): Tebakan awal untuk solusi. Jika None, akan diinisialisasi sebagai vektor nol.
        tolerance (float): Toleransi untuk kriteria konvergensi.
        max_iterations (int): Jumlah maksimum iterasi.

    Returns:
        np.array: Vektor solusi.
    """
    # Mendapatkan ukuran matriks
    n = len(b)

    # Inisialisasi tebakan awal jika tidak diberikan
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    # Iterasi untuk mencari solusi
    for k in range(max_iterations):
        x_old = x.copy()

        # Loop untuk setiap variabel (setiap baris)
        for i in range(n):
            # Hitung sigma(A[i,j] * x[j]) untuk j != i
            sigma1 = np.dot(A[i, :i], x[:i])
            sigma2 = np.dot(A[i, i + 1:], x_old[i + 1:])

            # Hitung nilai x[i] yang baru
            x[i] = (b[i] - sigma1 - sigma2) / A[i, i]

        # Cetak hasil setiap iterasi
        print(f"Iterasi {k + 1}: x = {x[0]:.4f}, y = {x[1]:.4f}, z = {x[2]:.4f}")

        # Cek konvergensi: jika perubahan solusi lebih kecil dari toleransi, hentikan iterasi
        if np.allclose(x, x_old, atol=tolerance):
            print("\nKonvergensi tercapai!")
            break

    return x


# --- PENDEFINISIAN MASALAH ---
# Berdasarkan susunan ulang persamaan untuk konvergensi yang lebih baik:
# 1. 3x + 2y + 2z = 25000
# 2. x + 2y + z = 15000
# 3. 2x + y + 3z = 20000

A = np.array([
    [3.0, 2.0, 2.0],
    [1.0, 2.0, 1.0],
    [2.0, 1.0, 3.0]
])

b = np.array([25000.0, 15000.0, 20000.0])

# --- EKSEKUSI DAN HASIL ---
print("--- Memulai Metode Gauss-Seidel ---")
# Menjalankan fungsi dengan tebakan awal x=0, y=0, z=0
solusi = gauss_seidel(A, b)
print("\n--- Hasil Akhir ---")
print(f"Solusi yang ditemukan:")
print(f"x = {solusi[0]:.4f}")
print(f"y = {solusi[1]:.4f}")
print(f"z = {solusi[2]:.4f}")