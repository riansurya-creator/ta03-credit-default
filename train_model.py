"""TA-03: Training model Regresi Logistik untuk prediksi gagal bayar kartu kredit.

Pemakaian:
    python train_model.py                      # memakai data/UCI_Credit_Card.csv
    python train_model.py path/ke/dataset.csv
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/UCI_Credit_Card.csv"
NUM_FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1"]
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
CAT_LEVELS = [[1, 2], [1, 2, 3, 4], [1, 2, 3]]
THRESHOLD = 0.5

# ------------------------------------------------------------------ 1. Load & cleaning
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"default.payment.next.month": "default",
                        "default payment next month": "default",
                        "PAY_1": "PAY_0"})
df = df.drop(columns=[c for c in ["ID"] if c in df.columns]).drop_duplicates()
print("Ukuran data:", df.shape)
print("Missing value per kolom (total):", int(df.isna().sum().sum()))
for col in NUM_FEATURES:
    df[col] = df[col].fillna(df[col].median())
for col in CAT_FEATURES:
    df[col] = df[col].fillna(df[col].mode()[0])
# Nilai kategori di luar dokumentasi dataset digabung ke kategori "lainnya"
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# ------------------------------------------------------------------ 2. EDA singkat
counts = df["default"].value_counts().sort_index()
print("\nDistribusi target:\n", counts, f"\nRasio gagal bayar: {counts[1] / counts.sum():.2%}")
plt.figure(figsize=(5, 4))
sns.countplot(x="default", data=df)
plt.xticks([0, 1], ["Tidak Gagal Bayar", "Gagal Bayar"])
plt.title("Distribusi Target (sebelum penyeimbangan)")
plt.tight_layout(); plt.savefig("figures/01_distribusi_target.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[NUM_FEATURES + ["default"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korelasi Fitur Numerik"); plt.tight_layout()
plt.savefig("figures/02_korelasi.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 3. Split + scaling
X = df[NUM_FEATURES + CAT_FEATURES]
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(categories=CAT_LEVELS, handle_unknown="ignore"), CAT_FEATURES),
])
X_train_t = preprocessor.fit_transform(X_train)   # fit hanya pada data latih
X_test_t = preprocessor.transform(X_test)

# ------------------------------------------------------------------ 4. Class imbalance
if HAS_SMOTE:
    X_train_t, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_t, y_train)
    class_weight = None
    print("\nSMOTE diterapkan pada data latih saja:", dict(pd.Series(y_train_bal).value_counts()))
else:
    y_train_bal = y_train
    class_weight = "balanced"
    print("\n[PERINGATAN] imbalanced-learn belum terpasang -> memakai class_weight='balanced'."
          "\nJalankan: pip install imbalanced-learn")

# ------------------------------------------------------------------ 5. Training
model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
model.fit(X_train_t, y_train_bal)

# ------------------------------------------------------------------ 6. Evaluasi
proba = model.predict_proba(X_test_t)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
print("\n--- Classification Report (threshold = %.2f) ---" % THRESHOLD)
print(classification_report(y_test, pred, target_names=["Tidak Gagal Bayar", "Gagal Bayar"]))
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
auc = roc_auc_score(y_test, proba)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  | AUC={auc:.4f}")

print("\n--- Dampak perubahan threshold ---")
for th in (0.3, 0.4, 0.5, 0.6):
    p = (proba >= th).astype(int)
    t_n, f_p, f_n, t_p = confusion_matrix(y_test, p).ravel()
    print(f"threshold {th:.1f}: precision={t_p / max(t_p + f_p, 1):.3f} "
          f"recall={t_p / (t_p + f_n):.3f}  FP={f_p}  FN={f_n}")

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["Tidak Gagal Bayar", "Gagal Bayar"]).plot(
    cmap=plt.cm.Blues, ax=ax, colorbar=False)
ax.set_title("Confusion Matrix"); plt.tight_layout()
plt.savefig("figures/03_confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="orange", label=f"Kurva ROC (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "--", color="darkblue", label="Tebakan Acak")
plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("True Positive Rate (TPR)")
plt.title("Kurva ROC"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("figures/04_roc_curve.png", dpi=150); plt.close()

# Koefisien (interpretasi bisnis)
names = preprocessor.get_feature_names_out()
coef = pd.Series(model.coef_[0], index=names).sort_values()
print("\n--- Koefisien model (terurut) ---\n", coef.round(3))

# ------------------------------------------------------------------ 7. Simpan
joblib.dump(model, "model/logreg_model.joblib")
joblib.dump(preprocessor, "model/scaler_preprocessor.joblib")
meta = {"num_features": NUM_FEATURES, "cat_features": CAT_FEATURES, "threshold": THRESHOLD,
        "metrics": {"auc": round(float(auc), 4),
                    "recall": round(tp / (tp + fn), 4),
                    "precision": round(tp / max(tp + fp, 1), 4),
                    "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4)},
        "smote": HAS_SMOTE}
json.dump(meta, open("model/meta.json", "w"), indent=2)
print("\nModel, preprocessor (StandardScaler + OneHot), dan meta.json tersimpan di folder model/")
