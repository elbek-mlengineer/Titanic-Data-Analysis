# -*- coding: utf-8 -*-
"""Titanic: Scaling, Encoding, Feature Engineering.ipynb"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Ma'lumotlarni yuklash
data = pd.read_csv('/content/train (1).csv')

# --- Dastlabki tahlil (EDA) ---
data.info()
data.describe()

# Tirik qolish ko'rsatkichlari (0 - vafot etgan, 1 - tirik qolgan)
all_survived = data['Survived'].value_counts(normalize=True)
print(f"Omon qolmaganlar: {all_survived[0]*100:.1f} %")
print(f"Tiriklar: {all_survived[1]*100:.1f} %")

# --- Missing Values (Yetishmayotgan qiymatlar) ---
# Ko'p qismi bo'sh bo'lgan 'Cabin' ustunini o'chirish
data = data.drop('Cabin', axis=1)

# 'Age' bo'sh joylarini median bilan to'ldirish
data['Age'] = data['Age'].fillna(data['Age'].median())

# 'Embarked' dagi bo'sh 2 ta qatorni eng ko'p uchraydigan port (mode) bilan to'ldirish (qatorni o'chirmaslik yaxshiroq)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Agar Fare da bo'sh qiymat bo'lsa (test to'plamida uchraydi)
data['Fare'] = data['Fare'].fillna(data['Fare'].median())


# --- Feature Engineering (Yangi belgilar yaratish) ---
# 1. Ismlardan unvonlarni ajratish (Mr, Mrs, Miss, Master va boshqalar)
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# 2. Oila a'zolari soni va yolg'iz sayohat qilganlik belgisi
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
data['Is_Alone'] = (data['Family_Size'] == 1).astype(int)

# 3. Yosh toifalari (Age bins)
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)

# 4. Outlier'lar: Fare qiymatidagi keskin farqni kamaytirish uchun log-transformatsiya
data['Fare_Log'] = np.log1p(data['Fare'])


# --- Encoding (Kategoriyalarni raqamga o'tkazish) ---
# Jinsni kodlash (male=1, female=0)
le = LabelEncoder()
data['Sex_encoded'] = le.fit_transform(data['Sex'])

# 'Embarked' va 'Title' uchun One-Hot Encoding
data = pd.get_dummies(data, columns=['Embarked', 'Title'], drop_first=True, dtype=int)


# --- Scaling (Raqamli ustunlarni bir xil masshtabga keltirish) ---
scaler = StandardScaler()
data[['Age_scaled', 'Fare_scaled']] = scaler.fit_transform(data[['Age', 'Fare_Log']])


# --- Tozalash: Modelga kiritilmaydigan ortiqcha ustunlarni tashlash ---
drop_cols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Age', 'Fare', 'Fare_Log']
data_clean = data.drop(columns=drop_cols)

# Tayyor dataset holatini tekshirish
print(data_clean.info())
data_clean.head()