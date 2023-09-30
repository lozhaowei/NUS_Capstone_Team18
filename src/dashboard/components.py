import streamlit as st
from streamlit_star_rating import st_star_rating

from src.dashboard.database import insert_model_feedback

def user_feedback_component(model):
    with st.form("feedback_form", clear_on_submit=True):
        st.write('User Feedback')

        rating = st_star_rating("Model rating", maxValue=5, defaultValue=5, key="rating",
                                customCSS="h3 {font-size: 14px;}")

        feedback = st.text_area('Add feedback', placeholder='Enter User Feedback')

        submitted = st.form_submit_button("Submit")

        if submitted:
            if feedback == '':
                st.warning('Please enter feedback before submitting!')
            else:
                # TODO add user id
                if insert_model_feedback({'feedback': feedback, 'rating': rating, 'model': model}) == 0:
                    st.success('Feedback submitted!')
                else:
                    st.warning('Error submitting feedback!')
