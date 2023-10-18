import streamlit as st

from src.dashboard.data.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components import summary_metrics_component, historical_retraining_data_visualisation_component, \
    real_time_data_visualisation_component, user_feedback_component, model_rating_component, generate_pdf_component

st.set_page_config(layout="wide")
if (
    st.session_state["authenticated"]
    and "Admin" in st.session_state["user_cognito_groups"]
):
    data = get_dashboard_data("convo").reset_index(drop=True)
    model_list = data['model'].unique()

    col1, col2 = st.columns([0.7, 0.3])
    col1.title('Conversations')
    models = col2.multiselect('Model', options=model_list, default=model_list[0])
    filtered_data = filter_data(data, models)
    elements = st.multiselect("Choose the elements that you want to show",
                            options=["Summary Metrics", "Real Time Data", "Historical Chart", "User Feedback"])

    st.divider()

    if "Summary Metrics" in elements:
        summary_metrics_component(filtered_data, models)
        st.divider()

    if "Real Time Data" in elements:
        real_time_data_visualisation_component()
        st.divider()

    if "Historical Chart" in elements:
        historical_chart = historical_retraining_data_visualisation_component(filtered_data, models)
        st.divider()

    # user feedback
    if "User Feedback" in elements:
        model_rating_component('video')
        user_feedback_component('video', model_list)

    # generate_pdf_component(filtered_data, models, historical_chart)

else:
    if st.session_state["authenticated"]:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")


