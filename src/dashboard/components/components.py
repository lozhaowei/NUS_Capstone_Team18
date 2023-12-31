import streamlit as st
from streamlit_star_rating import st_star_rating

from src.dashboard.components.data_handling import get_summary_metric_for_model, get_graph_for_summary_metric, \
    get_comparison_dates_for_summary_metrics, get_graph_for_real_time_component
from src.dashboard.data.database import insert_model_feedback, get_model_ratings, get_upvote_percentage_for_day
from src.dashboard.user_authen.authenticate_components import get_user_id
from src.dashboard.data.spark_pipeline import SparkPipeline


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
    Includes button to update latest 3 days of data
    :param entity: entity being recommended (video / convo)
    :param filtered_data: placeholder for function_mapping
    :param models: placeholder for function_mapping
    :return: like percentage graph
    """
    try:
        table_name = "nus_rs_video_upvote" if entity == "video" else "nus_rs_conversation_like"
        interacted_entity = "upvoted_videos" if entity == "video" else "liked_conversations"
        interacted_pct = "upvote_percentage" if entity == "video" else "like_percentage"
        data = get_upvote_percentage_for_day(table_name, interacted_entity, interacted_pct)
        st.subheader("Visualisation of Recommendations Generated")

        col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
        start_date = col1.date_input("Start Date", value=min(data["recommendation_date"]),
                                     min_value=min(data["recommendation_date"]),
                                     max_value=max(data["recommendation_date"]))
        end_date = col2.date_input("End Date", value=max(data["recommendation_date"]),
                                   min_value=min(data["recommendation_date"]),
                                   max_value=max(data["recommendation_date"]))

        available_metrics = [column for column in data.columns if column != "recommendation_date" and column != "dt"]
        columns = col3.multiselect("Metrics", options=available_metrics,
                                   default=available_metrics[0])

        data = data[(data["recommendation_date"] >= start_date) & (data["recommendation_date"] <= end_date)]

        refresh_button = st.button('Update latest 3 days of data')
        st.caption('This can take up to 10 minutes depending on data volume')

        if refresh_button:
            spark_pipeline = SparkPipeline()
            spark_pipeline.initialize_spark_session()

            if entity == 'video':
                spark_pipeline.run_video_upvote_percentage_pipeline()
            if entity == 'convo':
                spark_pipeline.run_conversation_like_percentage_pipeline()

            spark_pipeline.close_spark_session()

        tab1, tab2 = st.tabs(["Visualisation", "Data"])

        with tab1:
            for column in columns:
                fig = get_graph_for_real_time_component(data, column)
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)

                three_day_avg = data.tail(3)[column].mean()
                formatted_three_day_avg = f"{three_day_avg:,.2f}" if "percentage" not in column \
                    else f"{three_day_avg:.2%} "
                one_week_avg = data.tail(7)[column].mean()
                formatted_one_week_avg = f"{one_week_avg:,.2f}" if "percentage" not in column \
                    else f"{one_week_avg:.2%}"
                one_month_avg = data.tail(31)[column].mean()
                formatted_one_month_avg = f"{one_month_avg:,.2f}" if "percentage" not in column \
                    else f"{one_month_avg:.2%}"

                col1.metric("3D Average", formatted_three_day_avg)
                col2.metric("1W Average", formatted_one_week_avg)
                col3.metric("1M Average", formatted_one_month_avg)

        with tab2:
            st.dataframe(data)
        
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
            show_retraining = st.checkbox("Show Retraining Dates")

            fig = get_graph_for_summary_metric(filtered_data, freq, models, metrics, show_retraining)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(filtered_data)

        return fig

    except Exception as e:
        print('Error loading historical retraining data visualisation component: ', e)

def model_rating_component(entity):
    """
    Model Rating Component
    :param entity: item that the model recommended
    """
    try:
        st.subheader("Overall Model Rating")
        model_ratings = get_model_ratings(entity)

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

def user_feedback_component(entity, model_list):
    """
    User Feedback Form
    :param entity: item that the model recommended
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
                    return

                if 'username' and 'role' in st.session_state:
                    if insert_model_feedback({'feedback': feedback, 'rating': rating, 'model': model,
                                              'recommended_item': entity,
                                              'user_id': get_user_id(st.session_state.username),
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
