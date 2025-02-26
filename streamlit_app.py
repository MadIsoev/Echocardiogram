import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Echocardiogram')

st.write('Анализ набора данных Echocardiogram')

df = pd.read_csv("echocardiogram.data", sep=",", header=None, na_values="?", on_bad_lines="skip")

with st.expander('Data'):
  st.write("X")
  X_raw = df.drop('species', axis=1)
  st.dataframe(X_raw)

  st.write("y")
  y_raw = df.species
  st.dataframe(y_raw)









