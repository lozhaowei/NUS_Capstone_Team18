import streamlit as st

from user_authen.authenticate_components import check_login_status
from src.dashboard.data.data_handling import filter_data
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components import summary_metrics_component, historical_retraining_data_visualisation_component, \
      real_time_data_visualisation_component, user_feedback_component, model_rating_component, generate_pdf_component, \
      function_mapping

st.set_page_config(layout="wide")

check_login_status()

if st.session_state.role == "admin":

    data = get_dashboard_data("video").reset_index(drop=True)
    model_list = data['model'].unique()

    col1, col2 = st.columns([0.7, 0.3])
    col1.title('Videos')
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
        mapping[element]("video", filtered_data, models)
            
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


            
        

else:
    if st.session_state.username is not None:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")


