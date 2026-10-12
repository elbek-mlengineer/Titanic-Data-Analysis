# -*- coding: utf-8 -*-
"""
Titanic Dataset: Logistic Regression orqali tasniflash (Classification) modeli.
Ushbu skriptda ma'lumotlarni tayyorlash, belgilar muhandisligi (feature engineering),
nazariy asoslar (Sigmoid funksiyasi), model o'qitish va baholash bosqichlari jamlangan.
"""

# ==============================================================================
# 1. KUTUBXONALARNI IMPORT QILISH
# ==============================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 2. MA'LUMOTLARNI YUKLASH VA DASTLABKI KO'ZDAN KECHIRISH
# ==============================================================================
# Fayl yo'li (agar lokal papkada bo'lsa 'clean_titanic.csv', aks holda Colab yo'li)
file_path = "clean_titanic.csv" if os.path.exists("clean_titanic.csv") else "/content/clean_titanic.csv"
df = pd.read_csv(file_path)

# Jadval o'lchamini (qatorlar va ustunlar soni) chiqarish
print(f"Dataset o'lchami: {df.shape}")

# ==============================================================================
# 3. CHIZIQLI REGRESSIYANING TASNIFLASHDAGI CHEKLANISHI (EKSPERIMENT)
# ==============================================================================
# 'Fare' (chipta narxi) orqali omon qolishni bashorat qilish
X_fare_demo, y_demo = df[['Fare']], df['Survived']
lr = LinearRegression().fit(X_fare_demo, y_demo)

# Chipta narxi 0, 200 va 512 bo'lgan holatlar uchun bashorat:
# (Eslatma: Linear Regression [0, 1] ehtimollik oralig'idan chiqib ketishi mumkin)
test_fares = pd.DataFrame([[0], [200], [512]], columns=['Fare'])
print("Linear Regression bashoratlari:\n", lr.predict(test_fares))

# ==============================================================================
# 4. NAZARIY TUSHUNCHALAR: SIGMOID FUNKSIYASI VA Z QIYMATI
# ==============================================================================
# Kiruvchi sonni 0 va 1 oralig'idagi ehtimollikka o'tkazuvchi Sigmoid funksiyasi
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("\n--- Sigmoid qiymatlari ---")
print(f"z=-2  => {sigmoid(-2):.3f}")
print(f"z=0   => {sigmoid(0):.3f}")
print(f"z=0.5 => {sigmoid(0.5):.3f}")
print(f"z=2   => {sigmoid(2):.3f}")

# Chiziqli kombinatsiya: z = w * x + b
x_sym = sp.symbols('x')


def z(w, b, x):
    return w * x + b


print(f"w = 2.5, b = -1.4, x = 0 -> {z(2.5, -1.4, 0)}")
print(f"w = 2.5, b = -1.4, x = 1 -> {z(2.5, -1.4, 1)}")

# Sigmoid funksiyasining S-simon grafigini chizish
nums = np.linspace(-6, 6, 13)
print("\nSigmoid nuqtalari:", nums)

plt.figure(figsize=(7, 4))
plt.plot(nums, [sigmoid(val) for val in nums], marker='o', color='purple')
plt.grid(True)
plt.title("Sigmoid funksiyasi grafigi")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.show()

# ==============================================================================
# 5. MA'LUMOTLARNI TOZALASH VA BELGILARNI KODLASH (PREPROCESSING)
# ==============================================================================
# Bo'sh (NaN) qiymatlarni to'ldirish
df['Age'] = df['Age'].fillna(df['Age'].median())
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna('S')

# 'Sex' (jins) ustunini raqamga o'tkazish: male -> 0, female -> 1
if df['Sex'].dtype == object:
    df['Sex'] = df['Sex'].map({"male": 0, "female": 1})

# Embarked - 3 qiymat, one-hot
if 'Embarked' in df.columns and ('Embarked_Q' not in df.columns or 'Embarked_S' not in df.columns):
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Agar 'Embarked_Q' va 'Embarked_S' bool (True/False) bo'lsa, ularni 1 va 0 ga o'tkazamiz
for col in ['Embarked_Q', 'Embarked_S']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Takroriy ustunlar paydo bo'lmasligi uchun tozalash
df = df.loc[:, ~df.columns.duplicated()]

print("\nTozalangan ma'lumotlarning dastlabki 5 qatori:")
print(df.head())

# Yangi belgilar yaratish (Feature Engineering)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ==============================================================================
# 6. 'FARE' USTUNI BO'YICHA BIR BELGILI MODEL TAHLILI
# ==============================================================================
X_fare = df[['Fare']]
y_fare = df['Survived']
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fare, y_fare, test_size=0.2, random_state=42
)

# Chipta narxini standartlashtirish (o'rtacha=0, dispersiya=1)
sc_fare = StandardScaler().fit(X_train_f)
X_train_s = sc_fare.transform(X_train_f)
X_test_s = sc_fare.transform(X_test_f)

# Yo'lovchilarning yosh toifalari bo'yicha taqsimotini tekshirish
print("\nYosh toifalari taqsimoti:")
print(f"0 < age <= 15  -> {len(df[(df.Age > 0) & (df.Age <= 15)])}")
print(f"15 < age <= 30 -> {len(df[(df.Age > 15) & (df.Age <= 30)])}")
print(f"30 < age <= 45 -> {len(df[(df.Age > 30) & (df.Age <= 45)])}")
print(f"45 < age       -> {len(df[df.Age > 45])}")

# ==============================================================================
# 7. LOGISTIK REGRESSIYA MODELI (8 TA BELGI BILAN)
# ==============================================================================
feats_8 = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "Embarked_Q", "Embarked_S"]
y = df["Survived"]

# Stratifikatsiya orqali sinflar mutanosibligini saqlab train/test ga ajratish
Xtr_8, Xte_8, ytr_8, yte_8 = train_test_split(
    df[feats_8], y, test_size=0.2, random_state=42, stratify=y
)

# Standartlashtirish va modelni o'qitish
sc_8 = StandardScaler().fit(Xtr_8)
m_8 = LogisticRegression(max_iter=1000)
m_8.fit(sc_8.transform(Xtr_8), ytr_8)

Xte_8_scaled = sc_8.transform(Xte_8)
print(f"\nAccuracy (8 ta belgi bilan): {m_8.score(Xte_8_scaled, yte_8):.4f}")

print("\nPredictions (Dastlabki 5 ta):")
print(m_8.predict(Xte_8_scaled)[:5])

print("\nProbabilities (Dastlabki 2 ta):")
print(m_8.predict_proba(Xte_8_scaled)[:2])

# ==============================================================================
# 8. SODDALASHTIRILGAN MODEL (3 TA ASOSIY BELGI BILAN)
# ==============================================================================
feats = ["Pclass", "Sex", "Age"]

Xtr, Xte, ytr, yte = train_test_split(
    df[feats], y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler().fit(Xtr)
m = LogisticRegression(max_iter=1000)
m.fit(sc.transform(Xtr), ytr)
print(f"\nAccuracy (3 ta belgi bilan): {m.score(sc.transform(Xte), yte):.4f}")

# ==============================================================================
# 9. YANGI SHAXS UCHUN INDIVIDUAL BASHORAT QILISH
# ==============================================================================
person = {
    "Pclass": 3,
    "Sex": 0,
    "Age": 20
}

person_df = pd.DataFrame([person])[feats]
person_scaled = sc.transform(person_df)

prediction = m.predict(person_scaled)[0]
probabilities = m.predict_proba(person_scaled)[0]

status = "Tirik qolgan (Survived)" if prediction == 1 else "Vafot etgan (Not Survived)"
print("\n--- Yangi yo'lovchi tahlili ---")
print(f"Bashorat: {status}")
print(f"Tirik qolish ehtimoli: {probabilities[1] * 100:.2f}%")

# ==============================================================================
# 10. MODELNI METRIKALAR BO'YICHA BAHOLASH (EVALUATION)
# ==============================================================================
# Model 3 ta ustunga moslangan, shuning uchun test ma'lumotlarini qayta masshtablaymiz
Xte_scaled = sc.transform(Xte)
pred = m.predict(Xte_scaled)

print("\nConfusion Matrix (Haqiqiy model):")
print(confusion_matrix(yte, pred))

print("\nClassification Report:")
print(classification_report(yte, pred))
# FP qimmat --> Precision'ga e'tibor bering.
# FN qimmat --> Recall'ga e'tibor bering.
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)

# "Hamma o'ladi" baseline (barcha bashoratlar 0)
baseline_pred = np.zeros_like(yte)

print("\nBaseline Confusion Matrix:")
print(confusion_matrix(yte, baseline_pred))
print("\nBaseline Classification Report:")
print(classification_report(yte, baseline_pred, zero_division=0))

# Logistic Regression modeli: Tirik qolganlar (1 class) uchun Recall 0.69 (yoki ~69%).
# Ya'ni haqiqatda tirik qolganlarning 69% ini modelimiz to'g'ri topa olgan.
# "Hamma o'ladi" baseline modeli: Tirik qolganlar (1 class) uchun Recall 0.00 (0%).
# Chunki bu model hech kimni tirik qoldi deb topa olmaydi (barcha True Positive-lar 0 ga teng).

# ==============================================================================
# 11. THRESHOLD (CHEGARA) O'ZGARISHIGA NISBATAN PRECISION VA RECALL GRAFIGI
# ==============================================================================
# Ehtimolliklarni olish (tirik qolish sinfi uchun: index 1)
probs = m.predict_proba(Xte_scaled)[:, 1]

thresholds = np.linspace(0.1, 0.9, 50)
precisions = []
recalls = []

for t in thresholds:
    preds_t = (probs >= t).astype(int)
    precisions.append(precision_score(yte, preds_t, zero_division=0))
    recalls.append(recall_score(yte, preds_t, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', color='blue', linewidth=2)
plt.plot(thresholds, recalls, label='Recall', color='orange', linewidth=2)
plt.title("Threshold o'zgarishiga nisbatan Precision va Recall", fontsize=14)
plt.xlabel("Chegara (Threshold)", fontsize=12)
plt.ylabel("Ko'rsatkich Qiymati", fontsize=12)
plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Chegara (0.5)')
plt.grid(True)
plt.legend(fontsize=11)
plt.show()