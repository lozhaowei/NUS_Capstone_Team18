import streamlit as st

from src.dashboard.data.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components import summary_metrics_component, historical_retraining_data_visualisation_component, \
    user_feedback_component, model_rating_component, generate_pdf_component

st.set_page_config(layout="wide")

data = get_dashboard_data("convo").reset_index(drop=True)
model_list = data['model'].unique()

col1, col2 = st.columns([0.7, 0.3])
col1.title('Conversations')
models = col2.multiselect('Model', options=model_list, default=model_list[0])
filtered_data = filter_data(data, models)

st.divider()

summary_metrics_component(filtered_data, models)

st.divider()

historical_chart = historical_retraining_data_visualisation_component(filtered_data, models)

# user feedback
model_rating_component('video')
user_feedback_component('video', model_list)

generate_pdf_component(filtered_data, models, historical_chart)
