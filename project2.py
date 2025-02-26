import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Заголовок приложения
st.title('Анализ данных Echocardiogram')

# Загрузка данных
data_url = "https://raw.githubusercontent.com/MadIsoev/Echocardiogram/master/echocardiogram.data"
df = pd.read_csv(data_url, sep=",", header=None, na_values="?", on_bad_lines="skip")

# Определение названий столбцов
column_names = [
    "survival", "still-alive", "age", "pericardial-effusion", "fractional-shortening",
    "epss", "lvdd", "wall-motion-score", "wall-motion-index", "mult", "name",
    "group", "alive-at-1"
]
df.columns = column_names

# Отображение первых строк датасета
st.subheader("Первые строки датасета")
st.write(df.head())

# Обработка данных
st.subheader("Предобработка данных")

# Удаление строк с отсутствующими значениями в целевой переменной
df = df.dropna(subset=["still-alive"])
X = df.drop(columns=["still-alive", "name", "group"])
y = df["still-alive"].astype(int)

# Заполнение пропусков средними значениями
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Вывод метрик модели
st.subheader("Результаты модели")
st.write(f"Точность модели: {accuracy_score(y_test, y_pred):.2f}")
st.text("Отчет о классификации:")
st.text(classification_report(y_test, y_pred))

# Визуализация важности признаков
st.subheader("Важность признаков")
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
fig = px.bar(feature_importances.sort_values(by="Importance", ascending=False), x="Feature", y="Importance")
st.plotly_chart(fig)
