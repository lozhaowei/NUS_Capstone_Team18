import streamlit as st

from src.dashboard.data.data_handling import get_summary_metric_for_model, filter_data, \
    get_graph_for_summary_metric
from src.dashboard.data.database import get_dashboard_data
from src.dashboard.components import user_feedback_component, create_download_link

from fpdf import FPDF
from tempfile import NamedTemporaryFile


st.set_page_config(layout="wide")

data = get_dashboard_data("video").reset_index(drop=True)
model_list = data['model'].unique()

col1, col2 = st.columns(2)
col1.title('Videos')
models = col2.multiselect('Model', options=model_list, default=model_list[0])
filtered_data = filter_data(data, models)

st.divider()

precision_metric = get_summary_metric_for_model(filtered_data, models[0], 'Precision')
recall_metric = get_summary_metric_for_model(filtered_data, models[0], 'Recall')
f1_metric = get_summary_metric_for_model(filtered_data, models[0], 'F1 Score')

col1, col2, col3 = st.columns(3)
col1.metric("Precision", precision_metric[0], precision_metric[1])
col2.metric("Recall", recall_metric[0], recall_metric[1])
col3.metric("F1 Score", f1_metric[0], f1_metric[1])

st.divider()

if st.checkbox('Show metrics results'):
    st.write('Metrics Data')
    filtered_data

col1, col2, col3 = st.columns(3)
col1.subheader('Line chart')
freq = col2.radio("Frequency", ["D", "W", "M", "Y"], horizontal=True)
metrics = col3.multiselect('Metrics', options=['Precision', 'Recall', 'Accuracy', 'F1 Score', 'ROC AUC Score',
                                               'HitRatio@K', 'NDCG@K'], default='ROC AUC Score')

fig = get_graph_for_summary_metric(data, filtered_data, freq, models, metrics)
st.plotly_chart(fig, use_container_width=True)


# user feedback
user_feedback_component('video', model_list)

form = st.form("Report Generator")
submit = form.form_submit_button("Generate Report")

if submit:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B",20)
    pdf.cell(65, 10, txt=f'Precision: {str(precision_metric[0])}', align="C")
    pdf.cell(65, 10, txt=f'Recall: {str(recall_metric[0])}', align='C')
    pdf.cell(65, 10, txt=f'f1-score: {str(f1_metric[0])}', align='C')
    pdf.ln()
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.write_image(tmpfile.name)
        pdf.image(tmpfile.name, w=200, h=100)
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
    st.markdown(html, unsafe_allow_html=True)



