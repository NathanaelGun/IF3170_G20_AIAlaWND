# IF3170_G20_AIAlaWND

Repositori Tugas Besar 1 IF3170 Inteligensi Artifisial untuk mengoptimalkan
penempatan paket di dalam truk menggunakan algoritma local search. Program
memaksimalkan total value paket yang dimuat dengan memperhatikan ukuran truk,
kapasitas berat, overlap, support paket, dan larangan barang fragile menjadi
support.

Algoritma yang tersedia adalah Steepest Ascent Hill-Climbing, Hill-Climbing
with Sideways Move, Random Restart Hill-Climbing, Stochastic Hill-Climbing,
Simulated Annealing, dan Genetic Algorithm. Program juga mendukung multi-truk,
objective alternatif `eta_priority`, grafik eksperimen, serta visualisasi 3D
state awal dan akhir beserta paket di luar truk.

## Setup

Gunakan Python 3.10 atau lebih baru. Jalankan perintah berikut dari root
repositori untuk membuat virtual environment dan memasang dependensi.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependensi program tercantum di `requirements.txt`, yaitu Matplotlib.
Jendela visualisasi 3D juga memerlukan Tkinter yang tersedia pada instalasi
Python. Pada Windows, pastikan komponen Tcl/Tk ikut dipasang saat menginstal
Python.

## Menjalankan program

Jalankan program dari root repositori tanpa argumen atau flag:

```powershell
.\.venv\Scripts\python.exe src/main.py
```

Jika menggunakan lingkungan Python yang dependensinya sudah terpasang:

```powershell
python src/main.py
```

Ikuti menu interaktif berikut. Semua input harus diisi manual; input kosong
atau tidak valid akan diminta ulang.

1. Masukkan path file input JSON, misalnya `data/sample_input.json` untuk satu
   truk atau `data/sample_multi_truck.json` untuk beberapa truk.
2. Pilih objective `value` untuk total value paket, atau `eta_priority` untuk
   total `value / (1 + eta)` paket yang dimuat.
3. Pilih algoritma dan masukkan seed serta parameter yang diminta. HC meminta
   batas iterasi; ketik `tanpa` untuk HC selain Stochastic jika ingin tanpa
   batas iterasi. Sideways meminta maksimum total sideways move, Random Restart
   meminta maksimum restart, SA meminta temperatur awal, dan GA meminta jumlah
   populasi serta banyak generasi.
4. Setelah pencarian selesai, periksa state awal, state akhir, objective,
   durasi, dan statistik pencarian yang ditampilkan di terminal.
5. Masukkan lokasi penyimpanan hasil dengan ekstensi `.json`, misalnya
   `output/hasil.json`. Jawab `y` untuk membuat grafik, menyimpan gambar state
   3D, dan membuka jendela visualisasi interaktif, atau `n` untuk selesai.

Untuk eksperimen, pilih menu **4** untuk tiga run HC, **8** untuk tiga run SA,
atau **9** untuk 18 run GA. Masukkan tiga seed berbeda, konfigurasi eksperimen,
dan folder hasil, misalnya `output/experiments/hc`. Eksperimen GA terdiri dari
tiga variasi iterasi pada populasi tetap dan tiga variasi populasi pada iterasi
tetap, masing-masing diulang tiga kali. Hasil eksperimen meliputi JSON, grafik
PNG, dan rekapitulasi `summary.csv`.

Hasil JSON yang sudah disimpan dapat dibuka kembali sebagai visualisasi 3D:

```powershell
.\.venv\Scripts\python.exe -m src.visualization output/hasil.json
```

Untuk mengekspor gambar tanpa membuka jendela:

```powershell
.\.venv\Scripts\python.exe -m src.visualization output/hasil.json --save output/hasil_states_3d.png --no-show
```

## Pembagian tugas

| NIM | Nama | Tanggung jawab |
| --- | --- | --- |
| 13524054 | Wildan Abdurrahman Ghazali | Model data, representasi state, validasi constraints, objective, inisialisasi random, neighborhood, serta pembacaan dan penyimpanan JSON. Implementasi Hill-Climbing beserta variannya, menu dan integrasi program, eksperimen HC, serta penjelasan formulasi masalah dan metode HC dalam laporan. |
| 13524066 | Nathanael Gunawan | Implementasi Simulated Annealing, pencatatan probabilitas penerimaan dan stuck, serta visualisasi state awal dan akhir untuk semua algoritma. Eksperimen dan analisis SA, pembahasan visualisasi, serta perbandingan algoritma dan kesimpulan berdasarkan hasil seluruh anggota. |
| 13524132 | Dika Pramudya Nugraha | Implementasi Genetic Algorithm: populasi, seleksi, crossover, mutasi, dan penanganan individu invalid. Grafik eksperimen bersama, batch dan analisis 18 run GA, pengaruh jumlah populasi dan iterasi, serta tabel ringkasan hasil eksperimen. |
