import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Echocardiogramm')

st.write('Анализ набора данных Echocardiogramm')

df = pd.read_csv("echocardiogram.data", sep=",", header=None, na_values="?", on_bad_lines="skip")

columns = ["survival", "still-alive", "age-at-heart-attack", "pericardial-effusion",
           "partial-shortening", "epss", "lvdd", "wall-motion-score", "wall-motion-index",
           "mult", "name", "group", "alive-at"]

df.columns = columns

with st.expander('Datas'):
  st.write("X")
  X_raw = df.drop('still-alive', axis=1)
  st.dataframe(X_raw)

  st.write("y")
  y_raw = df["still-alive"]
  st.dataframe(y_raw)









