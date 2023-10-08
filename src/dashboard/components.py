import streamlit as st
from streamlit_star_rating import st_star_rating

from src.dashboard.data.database import insert_model_feedback

import base64


def user_feedback_component(recommended_item, model_list):
    """
    User Feedback Form
    :param recommended_item: item that the model recommended
    :param model_list: list of models
    """
    with st.form("feedback_form", clear_on_submit=True):
        st.subheader('User Feedback')

        rating = st_star_rating("Model rating", maxValue=5, defaultValue=5, key="rating",
                                customCSS="h3 {font-size: 14px;}")

        model = st.selectbox('Choose model', options=model_list)
        feedback = st.text_area('Add feedback', placeholder='Enter User Feedback', max_chars=500)

        submitted = st.form_submit_button("Submit")

        if submitted:
            if feedback == '':
                st.warning('Please enter feedback before submitting!')
            else:
                # TODO add user id
                if insert_model_feedback({'feedback': feedback, 'rating': rating, 'model': model,
                                          'recommended_item': recommended_item}) == 0:
                    st.success('Feedback submitted!')
                else:
                    st.warning('Error submitting feedback!')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'