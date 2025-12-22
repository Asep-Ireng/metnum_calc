# 🧮 Kalkulator Metode Numerik

Aplikasi kalkulator metode numerik berbasis **Streamlit** yang mengimplementasikan **20 metode** untuk menyelesaikan berbagai permasalahan matematika.

## 📋 Daftar Metode

| Kategori | Metode | Jumlah |
|----------|--------|--------|
| **Metode Akar** | Bisection, Regula Falsi, Newton-Raphson, Secant | 4 |
| **Interpolasi** | Newton Divided Difference, Lagrange | 2 |
| **Integrasi Numerik** | Trapezoidal, Simpson 1/3, Simpson 3/8 | 3 |
| **Diferensiasi Numerik** | Forward, Backward, Central | 3 |
| **Penyelesaian ODE** | Euler, Heun, Runge-Kutta 4 | 3 |
| **Sistem Persamaan Linear** | Gauss, Gauss-Jordan, LU, Jacobi, Gauss-Seidel | 5 |

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 3. Buka Browser
Akses aplikasi di `http://localhost:8501`

## 📁 Struktur Project

```
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependencies
├── methods/
│   ├── root_finding.py     # Metode pencarian akar
│   ├── interpolation.py    # Metode interpolasi
│   ├── integration.py      # Metode integrasi numerik
│   ├── differentiation.py  # Metode diferensiasi numerik
│   ├── ode.py              # Solver ODE
│   └── linear_systems.py   # Solver sistem linear
└── utils/
    └── parser.py           # Parser fungsi matematika
```

## ✨ Fitur

- **Input Interaktif** - Form input untuk setiap metode
- **Tabel Iterasi** - Menampilkan proses perhitungan step-by-step
- **Visualisasi Grafik** - Grafik konvergensi dan hasil
- **Analisis Error** - Perhitungan error relatif
- **Mode Perbandingan** - Bandingkan hasil antar metode

## 📝 Contoh Penggunaan

### Mencari Akar Persamaan
```
f(x) = x³ - x - 2
Interval: [1, 2]
Metode: Bisection
```

### Interpolasi
```
Data: x = [1, 2, 3, 4, 5], y = [1, 4, 9, 16, 25]
Target: x = 2.5
```

### Integrasi
```
f(x) = x²
Interval: [0, 1]
Metode: Simpson 1/3
```

## 👨‍💻 Author

Project Mata Kuliah **Metode Numerik**

## 📄 License

MIT License
