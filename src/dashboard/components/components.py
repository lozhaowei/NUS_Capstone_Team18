import streamlit as st
from streamlit_star_rating import st_star_rating

from src.dashboard.data.data_handling import get_summary_metric_for_model, get_comparison_dates_for_summary_metrics, \
    get_graph_for_summary_metric, get_graph_for_real_time_component
from src.dashboard.data.database import insert_model_feedback, get_model_ratings, get_upvote_percentage_for_day


def summary_metrics_component(entity, filtered_data, models):
    """
    Visualise the summary metrics for each model selected in the page.
    :param entity: recommended item
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
        st.markdown(f"Summary metrics shown are as of **{latest_date.date()}** and change in metrics "
                    f"is compared to **{second_latest_date.date()}**.")

def real_time_data_visualisation_component(entity, filtered_data, models):
    """
    Visualise total like percentage of each entity
    :param entity:
    :param filtered_data:
    :param models:
    :return:
    """
    try:
        table_name = "nus_rs_video_upvote" if entity == "video" else "nus_rs_conversation_like"
        data = get_upvote_percentage_for_day(table_name)
        st.subheader("Visualisation of Recommendations Generated")

        col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
        start_date = col1.date_input("Start Date", value=min(data["dt"]), min_value=min(data["dt"]),
                                     max_value=max(data["dt"]))
        end_date = col2.date_input("End Date", value=max(data["dt"]), min_value=min(data["dt"]),
                                   max_value=max(data["dt"]))

        available_metrics = [column for column in data.columns if column != "dt"]
        columns = col3.multiselect("Metrics", options=available_metrics,
                                   default=available_metrics[0])

        data = data[(data["dt"] >= start_date) & (data["dt"] <= end_date)]
        
        for column in columns:
            fig = get_graph_for_real_time_component(data, column)
            st.plotly_chart(fig, use_container_width=True)
        
        return fig

    except Exception as e:
        print('Error loading realtime data visualisation component: ', e)

def historical_retraining_data_visualisation_component(entity, filtered_data, models):
    """
    Plots the graph for historical retraining information of models.
    :param filtered_data: filtered dataframe based on the models selected for the page
    :param models: list of models selected for the page
    """
    try:
        tab1, tab2 = st.tabs(["Visualisation", "Data"])

        with tab1:
            col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
            col1.subheader('Summary Metrics in Historical Retraining')
            freq = col2.radio("Frequency", ["D", "W", "M", "Y"], horizontal=True)
            metrics = col3.multiselect('Metrics', options=['Precision', 'Recall', 'Accuracy', 'F1 Score',
                                                           'ROC AUC Score', 'HitRatio@K', 'NDCG@K'],
                                       default='ROC AUC Score')

            fig = get_graph_for_summary_metric(filtered_data, freq, models, metrics)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write(filtered_data)

        return fig

    except Exception as e:
        print('Error loading historical retraining data visualisation component: ', e)

def model_rating_component(recommended_item):
    """
    Model Rating Component
    :param recommended_item: item that the model recommended
    """
    try:
        st.subheader("Overall Model Rating")
        model_ratings = get_model_ratings(recommended_item)

        if model_ratings is None or len(model_ratings) == 0:
            print('Error getting model ratings')
            return

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

    except Exception as e:
        print('Error loading model rating component: ', e)

def user_feedback_component(recommended_item, model_list):
    """
    User Feedback Form
    :param recommended_item: item that the model recommended
    :param model_list: list of models
    """
    try:
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
                    # TODO add user id ?
                    if insert_model_feedback({'feedback': feedback, 'rating': rating, 'model': model,
                                              'recommended_item': recommended_item,
                                              'role':  st.session_state.role}) == 0:
                        st.success('Feedback submitted!')
                    else:
                        st.warning('Error submitting feedback!')

    except Exception as e:
        print('Error loading user feedback form: ', e)

def feedback_component(entity, models):
    model_rating_component(entity)
    user_feedback_component(entity, models)

def component_mapping():
    """
    Dictionary to allow easy component retrieval
    :return: component dictionary
    """
    component_map = {
        "Summary Metrics": summary_metrics_component,
        "Real Time Data": real_time_data_visualisation_component,
        "Historical Chart": historical_retraining_data_visualisation_component,
        "User Feedback": feedback_component
    }
    return component_map
