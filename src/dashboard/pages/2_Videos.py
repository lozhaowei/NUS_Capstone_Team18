import streamlit as st

from src.dashboard.data.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components import summary_metrics_component, historical_retraining_data_visualisation_component, \
      real_time_data_visualisation_component, user_feedback_component

st.set_page_config(layout="wide")

data = get_dashboard_data("video").reset_index(drop=True)
model_list = data['model'].unique()

col1, col2 = st.columns([0.7, 0.3])
col1.title('Videos')
models = col2.multiselect('Model', options=model_list, default=model_list[0])
filtered_data = filter_data(data, models)

st.divider()

summary_metrics_component(filtered_data, models)

st.divider()

real_time_data_visualisation_component()

st.divider()

historical_retraining_data_visualisation_component(filtered_data, models)

# user feedback
user_feedback_component('video', model_list)
