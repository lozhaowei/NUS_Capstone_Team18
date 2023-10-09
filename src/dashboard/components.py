import streamlit as st
from streamlit_star_rating import st_star_rating

from src.data import database
from src.video_recommend.knn import create_embedding_matrices
from src.dashboard.data.data_handling import get_summary_metric_for_model, get_comparison_dates_for_summary_metrics, get_graph_for_summary_metric
from src.dashboard.data.database import get_data_for_real_time_section_videos, insert_model_feedback

def summary_metrics_component(filtered_data, models):
    """
    Visualise the summary metrics for each model selected in the page.
    :param filtered_data: filtered dataframe based on the models selected for the page
    :param models: list of models selected for the page
    """
    for i in range(len(models)):
        precision_metric = get_summary_metric_for_model(filtered_data, models[i], 'Precision')
        recall_metric = get_summary_metric_for_model(filtered_data, models[i], 'Recall')
        f1_metric = get_summary_metric_for_model(filtered_data, models[i], 'F1 Score')
        latest_date, second_latest_date = get_comparison_dates_for_summary_metrics(filtered_data, models[i])

        st.subheader(f"Summary Metrics ({models[i]})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", precision_metric[0], precision_metric[1])
        col2.metric("Recall", recall_metric[0], recall_metric[1])
        col3.metric("F1 Score", f1_metric[0], f1_metric[1])
        st.markdown(f"Summary metrics shown are as of **{latest_date.date()}** and change in metrics is compared to **{second_latest_date.date()}**.")

def real_time_data_visualisation_component():
    data, video_data, season_data, vote_data, user_interest_data = get_data_for_real_time_section_videos("rs_daily_video_for_user")

    col1, col2, col3 = st.columns([0.65, 0.1, 0.25])
    col1.subheader("Latest Model Metrics")
    date = col2.selectbox("Select Date", data["created_at"].dt.date.unique())
    filtered_data = data[data["created_at"].dt.date == date]
    user = col3.selectbox("Select User", filtered_data["user_id"].unique())

    user_interest_matrix, video_category_matrix = create_embedding_matrices(user_interest_data, season_data, video_data, vote_data, date)
    user_interest = user_interest_matrix.loc[user]
    st.write(user_interest)


        
def historical_retraining_data_visualisation_component(filtered_data, models):
    """
    Plots the graph for historical retraining information of models.
    :param filtered_data: filtered dataframe based on the models selected for the page
    :param models: list of models selected for the page
    """
    if st.checkbox('Show metrics results'):
        st.write('Metrics Data')
        filtered_data

    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    col1.subheader('Summary Metrics in Historical Retraining')
    freq = col2.radio("Frequency", ["D", "W", "M", "Y"], horizontal=True)
    metrics = col3.multiselect('Metrics', options=['Precision', 'Recall', 'Accuracy', 'F1 Score', 'ROC AUC Score',
                                                'HitRatio@K', 'NDCG@K'], default='ROC AUC Score')

    fig = get_graph_for_summary_metric(filtered_data, freq, models, metrics)
    st.plotly_chart(fig, use_container_width=True)

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
