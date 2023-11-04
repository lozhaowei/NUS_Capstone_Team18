import streamlit as st

from user_authen.authenticate_components import check_login_status
from src.dashboard.data.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components.components import function_mapping

st.set_page_config(layout="wide")

check_login_status()

if st.session_state.role == "admin":
    data = get_dashboard_data("convo").reset_index(drop=True)
    model_list = data['model'].unique()

    col1, col2 = st.columns([0.7, 0.3])
    col1.title('Conversations')
    models = col2.multiselect('Model', options=model_list, default=model_list[0])
    filtered_data = filter_data(data, models)
    element_list = ["Summary Metrics", "Historical Chart", "User Feedback"]
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
        mapping[element]("video", filtered_data, models)

    # generate_pdf_component(filtered_data, models, historical_chart)

else:
    if st.session_state.username is not None:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")


