"""Aplikasi Flask: Prediksi Risiko Gagal Bayar Kartu Kredit (Regresi Logistik)."""
import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "model", "logreg_model.joblib"))
preprocessor = joblib.load(os.path.join(BASE, "model", "scaler_preprocessor.joblib"))
meta = json.load(open(os.path.join(BASE, "model", "meta.json")))
NUM, CAT, THRESHOLD = meta["num_features"], meta["cat_features"], meta["threshold"]

PAY_OPTIONS = [(-2, "-2 : Tidak ada tagihan"), (-1, "-1 : Lunas tepat waktu"), (0, "0 : Pakai kredit revolving"),
               (1, "1 : Telat 1 bulan"), (2, "2 : Telat 2 bulan"), (3, "3 : Telat 3 bulan"),
               (4, "4 : Telat 4 bulan"), (5, "5 : Telat 5 bulan"), (6, "6 : Telat 6 bulan"),
               (7, "7 : Telat 7 bulan"), (8, "8 : Telat 8 bulan atau lebih")]


def parse_form(form):
    """Ambil & validasi input; lempar ValueError berisi pesan yang ramah pengguna."""
    try:
        row = {
            "LIMIT_BAL": float(form["LIMIT_BAL"]), "AGE": int(form["AGE"]),
            "PAY_0": int(form["PAY_0"]), "PAY_2": int(form["PAY_2"]), "PAY_3": int(form["PAY_3"]),
            "BILL_AMT1": float(form["BILL_AMT1"]), "PAY_AMT1": float(form["PAY_AMT1"]),
            "SEX": int(form["SEX"]), "EDUCATION": int(form["EDUCATION"]), "MARRIAGE": int(form["MARRIAGE"]),
        }
    except (KeyError, ValueError):
        raise ValueError("Semua kolom wajib diisi dengan angka yang valid.")
    if row["LIMIT_BAL"] <= 0:
        raise ValueError("Batas kredit harus lebih dari 0.")
    if not 18 <= row["AGE"] <= 100:
        raise ValueError("Usia harus antara 18 dan 100 tahun.")
    if row["PAY_AMT1"] < 0:
        raise ValueError("Jumlah pembayaran tidak boleh negatif.")
    return row


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, request.form
    if request.method == "POST":
        try:
            row = parse_form(request.form)
            X = pd.DataFrame([row])[NUM + CAT]
            proba = float(model.predict_proba(preprocessor.transform(X))[0, 1])
            risky = proba >= THRESHOLD
            result = {"label": "Berisiko Gagal Bayar" if risky else "Tidak Berisiko",
                      "risky": risky, "proba": round(proba * 100, 1),
                      "threshold": int(THRESHOLD * 100)}
        except ValueError as e:
            error = str(e)
    return render_template("index.html", result=result, error=error, values=values,
                           pay_options=PAY_OPTIONS, metrics=meta["metrics"])


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
