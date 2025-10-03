import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

st.title("Telco Customer Churn Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(DATA_PATH)

st.write("Dataset Preview", df.head())

fig, ax = plt.subplots()
sns.countplot(x = 'Churn', data = df, ax = ax)

st.pyplot(fig)

st.bar_chart(df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0))
