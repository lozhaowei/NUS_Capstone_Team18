import streamlit as st
from streamlit_star_rating import st_star_rating

from src.dashboard.data.database import insert_model_feedback, get_model_ratings

def model_rating_component(recommended_item):
    """
    Model Rating Component
    :param recommended_item: item that the model recommended
    """
    st.subheader("Overall Model Rating")
    model_ratings = get_model_ratings(recommended_item)

    # maximum amount of columns per row
    max_columns = 3
    cols = st.columns(max_columns)
    counter = 0

    for k, v in model_ratings.items():
        # wraps columns around
        col = cols[counter % max_columns]
        col.metric(k, v)
        counter += 1

    # add spacing
    st.write("")

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
                # TODO add user id (and role?)
                if insert_model_feedback({'feedback': feedback, 'rating': rating, 'model': model,
                                          'recommended_item': recommended_item}) == 0:
                    st.success('Feedback submitted!')
                else:
                    st.warning('Error submitting feedback!')
