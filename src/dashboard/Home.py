import streamlit as st
from src.dashboard.data.data_handling import load_data

st.set_page_config(layout="wide")

st.write("Hello World")

metrics_data = load_data().reset_index(drop=True)
st.write(metrics_data)
