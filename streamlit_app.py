import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Echocardiogramm')

st.write('Анализ набора данных Echocardiogram')

df = pd.read_csv("echocardiogram.data", sep=",", header=None, na_values="?", on_bad_lines="skip")

columns = ["survival", "still-alive", "age-at-heart-attack", "pericardial-effusion",
           "partial-shortening", "epss", "lvdd", "wall-motion-score", "wall-motion-index",
           "mult", "name", "group", "alive-at"]

df.columns = columns

with st.expander('Data'):
  st.write("X")
  X_raw = df.drop('still-alive', axis=1)
  st.dataframe(X_raw)

  st.write("y")
  y_raw = df["still-alive"]
  st.dataframe(y_raw)

with st.sidebar:
  st.header("Введите признаки: ")
  island = st.selectbox('Island', ('Torgersen', 'Dream', 'Biscoe'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 44.5)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.3)
  flipper_length_mm = st.slider('Flipper length (mm)', 32.1, 59.6, 44.5)
  body_mass_g = st.slider('Body mass (g)', 32.1, 59.6, 44.5)
  gender = st.selectbox('Gender', ('female', 'male'))







