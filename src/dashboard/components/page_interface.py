import streamlit as st

from src.dashboard.components.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components.components import component_mapping

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
    mapping = component_mapping()

    remove_sidebar = st.checkbox("Remove Sidebar for Printing")

    if remove_sidebar:
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
