import streamlit as st
import pandas as pd


st.set_page_config(layout="wide")
st.title("Conversations")

col1, col2, col3 = st.columns(3)
col1.metric("Precision", "50%", "3%")
col2.metric("Recall", "50%", "-8%")
col3.metric("F1 Score", "30%", "4%")

st.divider()

# Replace with data pulled from database
metrics_data = pd.DataFrame({
        'precision': [0.3, 0.4, 0.5],
        'recall': [0.3, 0.5, 0.1],
        'f1 score': [0.3, 0.3, 0.3],
        'dt': ['2023-09-23', '2023-09-24', '2023-09-25']
    })


if st.checkbox('Show metrics results'):
    st.write('Metrics Data')
    metrics_data

st.subheader('Line chart')
st.line_chart(metrics_data, x='dt')
