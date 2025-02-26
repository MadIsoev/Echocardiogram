import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

st.title('💓 Echocardiogram Analysis')

st.write('Анализ данных и предсказание выживаемости')

data_url = "https://raw.githubusercontent.com/MadIsoev/Echocardiogram/master/echocardiogram.data"
df = pd.read_csv(data_url, sep=",", header=None, na_values="?", on_bad_lines="skip")

column_names = [
    "survival", "still-alive", "age", "pericardial-effusion", "fractional-shortening",
    "epss", "lvdd", "wall-motion-score", "wall-motion-index", "mult", "name",
    "group", "alive-at-1"
]
df.columns = column_names

with st.expander('📊 Data Overview'):
    st.write("**Feature Matrix (X)**")
    X_raw = df.drop(columns=["still-alive", "name", "group"], errors='ignore')
    st.dataframe(X_raw)
    
    st.write("**Target Variable (y)**")
    y_raw = df["still-alive"].astype(int)
    st.dataframe(y_raw)

with st.sidebar:
    st.header("🔧 Введите признаки: ")
    age = st.slider('Возраст', float(df.age.min()), float(df.age.max()), float(df.age.mean()))
    fractional_shortening = st.slider('Фракционное укорочение', float(df["fractional-shortening"].min()), float(df["fractional-shortening"].max()), float(df["fractional-shortening"].mean()))
    epss = st.slider('EPSS', float(df.epss.min()), float(df.epss.max()), float(df.epss.mean()))
    lvdd = st.slider('LVDD', float(df.lvdd.min()), float(df.lvdd.max()), float(df.lvdd.mean()))
    wall_motion_index = st.slider('Индекс движения стенки', float(df["wall-motion-index"].min()), float(df["wall-motion-index"].max()), float(df["wall-motion-index"].mean()))

data = {
    'age': age,
    'fractional-shortening': fractional_shortening,
    'epss': epss,
    'lvdd': lvdd,
    'wall-motion-index': wall_motion_index
}
input_df = pd.DataFrame(data, index=[0])
input_combined = pd.concat([input_df, X_raw], axis=0)

with st.expander('📥 Input Features'):
    st.write('**Selected Patient Features**')
    st.dataframe(input_df)
    st.write('**Combined Data (New Input + Original)**')
    st.dataframe(input_combined)

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
y = y_raw
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

input_df_aligned = pd.DataFrame(columns=X_raw.columns)
input_df_aligned = pd.concat([input_df_aligned, input_df], ignore_index=True).fillna(0)
df_input_scaled = pd.DataFrame(scaler.transform(input_df_aligned), columns=X_raw.columns)
prediction = clf.predict(df_input_scaled)
prediction_proba = clf.predict_proba(df_input_scaled)

df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Not Alive', 'Alive'])
st.subheader('🔮 Predicted Survival')
st.dataframe(df_prediction_proba, hide_index=True)

survival_status = np.array(['Not Alive', 'Alive'])
st.success(f"Predicted status: **{survival_status[prediction][0]}**")

# Data Visualization
st.subheader("📊 Data Visualization")

# Scatter Plot
fig1 = px.scatter(df, x='age', y='wall-motion-index', color='still-alive', title='Age vs. Wall Motion Index')
st.plotly_chart(fig1)

# Histogram
fig2 = px.histogram(df, x='age', nbins=30, title='Distribution of Age')
st.plotly_chart(fig2)

# Correlation Heatmap
st.subheader("🔎 Feature Correlations")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

st.write("💡 **Tip:** Use sidebar sliders to input patient details and get predictions!")




