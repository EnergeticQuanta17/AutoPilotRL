# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go

import pandas as pd

import optuna


def study_loader():
    study_name = "study-26"
    storage_name = f"sqlite:///C:/Users/mpree/Downloads/{study_name}.db"

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )

    return study

study = study_loader()

trials_df = study.trials_dataframe()

app = Dash(__name__)

for i in ['datetime_start', 'datetime_complete', 'duration']:
    trials_df[i] = trials_df[i].astype(str)

fig = go.Figure(
    data=[go.Scatter(x=trials_df['number'], y=trials_df['value'], mode='markers')],
    layout=go.Layout(
        title='Comparing different trials',
        xaxis=dict(title="Trial Number"),
        yaxis=dict(title="Reward Score")
    )
)

app.layout = html.Div([
    html.Div(children='My First Study Data'),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
