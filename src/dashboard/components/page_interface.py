import streamlit as st

from src.dashboard.data.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components.components import function_mapping

def page(entity, title):
    data = get_dashboard_data(entity).reset_index(drop=True)
    model_list = data['model'].unique()

    col1, col2 = st.columns([0.7, 0.3])
    col1.title(title)
    models = col2.multiselect('Model', options=model_list, default=model_list[0])
    filtered_data = filter_data(data, models)

    element_list = ["Summary Metrics", "Real Time Data", "Historical Chart", "User Feedback"]
    elements = st.multiselect("Choose the elements that you want to show",
                              options=element_list,
                              default=element_list)
    mapping = function_mapping()

    hide_sidebar = st.checkbox("Hide Sidebar")

    if hide_sidebar:
        st.markdown(
            """
            <style>
                [class="css-vk3wp9 eczjsme11"] {
                    display: none
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    for element in elements:
        st.divider()
        if element == "User Feedback":
            mapping[element](entity, model_list)
        else:
            mapping[element](entity, filtered_data, models)

        # if "Summary Metrics" in elements:
        #     summary_metrics_component(filtered_data, models)
        #     st.divider()

        # if "Real Time Data" in elements:
        #     real_time_data_visualisation_component()
        #     st.divider()

        # if "Historical Chart" in elements:
        #     historical_chart = historical_retraining_data_visualisation_component(filtered_data, models)
        #     st.divider()

        # # user feedback
        # if "User Feedback" in elements:
        #     model_rating_component('video')
        #     user_feedback_component('video', model_list)

        # generate_pdf_component(filtered_data, models, historical_chart)
