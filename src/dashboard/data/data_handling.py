import datetime

import pandas as pd
import plotly.express as px

def get_summary_metric_for_model(data, model, metric):
    model_data = data[data["model"] == model]
    model_data_latest_metric = model_data.nlargest(1, "dt").iloc[-1][metric].squeeze()
    model_data_previous_metric = model_data.nlargest(2, "dt").iloc[-1][metric].squeeze()

    pct_improvement = (model_data_latest_metric - model_data_previous_metric) / model_data_previous_metric

    return [round(model_data_latest_metric, 2), f"{round(100 * pct_improvement, 0)}%"]

def get_comparison_dates_for_summary_metrics(data, model):
    model_data = data[data["model"] == model]
    latest_date = model_data.nlargest(1, "dt").iloc[-1]["dt"]
    second_latest_date = model_data.nlargest(2, "dt").iloc[-1]["dt"]

    return latest_date, second_latest_date

def filter_data(data, models):
    mask = data["model"].isin(models)

    return data.copy()[mask]

def get_graph_for_real_time_component(data, column):
    line_chart_data = data[[column] + ["dt"]].set_index("dt")
    line_chart_data = line_chart_data.apply(pd.to_numeric)

    fig = px.line(line_chart_data, x=line_chart_data.index, y=column, title=f"{column} Visualisation")

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True, thickness=0.05)
    )

    fig.update_layout(
        autosize=True
    )

    return fig

def get_chart_data_for_multiple_models(data, models, metrics):
    new_df = pd.DataFrame(index=data["dt"].unique())
    filtered_data = data.copy()
    for model in models:
        suffix = f"_{model}"
        model_data = filtered_data[filtered_data["model"] == model].set_index("dt")
        model_data = model_data[metrics]
        model_data.columns = [col + suffix for col in model_data.columns]
        new_df = pd.concat([new_df, model_data], axis=1)

    return new_df

def get_graph_for_summary_metric(filtered_data, freq, models, metrics):
    if len(models) == 1:
        line_chart_data = filtered_data[metrics + ["dt"]].set_index("dt")
        columns_to_plot = metrics
    else:
        line_chart_data = get_chart_data_for_multiple_models(filtered_data, models, metrics)
        columns_to_plot = line_chart_data.columns

    line_chart_data = line_chart_data.resample(freq).mean()

    fig = px.line(line_chart_data, x=line_chart_data.index, y=columns_to_plot, title="Model Metrics")

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True)
    )

    if freq == "D":
        date_iterator = line_chart_data.index[0] # TODO: Change to the first retraining date applicable
        while date_iterator <= line_chart_data.index[-1]:
            fig.add_vline(x=date_iterator, line_dash="dash", line_color="red")
            date_iterator += datetime.timedelta(days=3)

    fig.update_layout(
        autosize=True
    )

    return fig
