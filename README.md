# TA-03: Prediksi Risiko Gagal Bayar Kartu Kredit (Regresi Logistik)

Rian Surya | NIM 301240058 | Kelas 5C

## Struktur
```
TA03_analisis.ipynb   notebook EDA, preprocessing, training, evaluasi
train_model.py        versi skrip dari notebook (menyimpan model + grafik)
app.py                aplikasi web Flask
templates/index.html  formulir input dan tampilan hasil
model/                logreg_model.joblib, scaler_preprocessor.joblib, meta.json (dibuat saat training)
data/                 UCI_Credit_Card.csv (unduh dari Kaggle)
requirements.txt, Procfile
```

## Langkah menjalankan
1. Unduh dataset **"Default of Credit Card Clients"** (UCI/Kaggle), simpan sebagai `data/UCI_Credit_Card.csv`.
2. Pasang library: `pip install -r requirements.txt`
3. Latih model: `python train_model.py` (atau jalankan `TA03_analisis.ipynb`). File di `model/` akan terbentuk.
4. Jalankan aplikasi: `python app.py`, lalu buka http://localhost:5000

## Deploy ke Render
1. Upload seluruh folder ke repositori GitHub, **termasuk folder `model/`** (hasil training harus ikut di-commit).
2. Di render.com: *New > Web Service*, hubungkan repositori.
3. Build command: `pip install -r requirements.txt`  |  Start command: `gunicorn app:app`
4. Pastikan versi `scikit-learn` di `requirements.txt` sama dengan yang dipakai saat training (cek dengan `pip show scikit-learn`, lalu tulis `scikit-learn==x.y.z`) agar file joblib terbaca tanpa error.

## Kerangka laporan
1. Pendahuluan dan latar belakang bisnis
2. Dataset (sumber, jumlah baris, deskripsi fitur, target)
3. EDA (distribusi target, hubungan PAY_0 dan LIMIT_BAL dengan gagal bayar)
4. Preprocessing (pembersihan kategori, one-hot, StandardScaler, SMOTE hanya pada data latih)
5. Model dan evaluasi (Confusion Matrix, Precision, Recall, F1, AUC, tabel dampak threshold)
6. Analisis bisnis: FP vs FN, mana yang lebih merugikan, justifikasi memilih Recall, interpretasi koefisien
7. Arsitektur aplikasi dan hasil deployment (tautan, screenshot)
8. Kesimpulan, keterbatasan, saran

## Kerangka video (5-7 menit)
Masalah bisnis (0:30) > data dan EDA (1:00) > preprocessing dan model (1:00) > hasil dan metrik (1:30) > demo aplikasi dengan 2 contoh nasabah, satu berisiko dan satu aman (1:30) > rekomendasi dan keterbatasan (0:45)
