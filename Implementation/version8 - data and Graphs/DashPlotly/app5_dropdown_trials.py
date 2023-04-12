# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px

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

for i in ['datetime_start', 'datetime_complete']:
    trials_df[i] = trials_df[i].apply(lambda x: int(x.timestamp()))

trials_df['duration'] = trials_df['duration'].dt.total_seconds()

app.layout = html.Div([
    html.Div([
        html.Label('Select column:'),
        dcc.Dropdown(trials_df.columns.to_list(), 'value', id='dropdown-selection'),
    ], style={'width': '20%', 'float': 'left'}),


    html.Div([
        dcc.Graph(id='graph-content'),
    ], style={'width': '80%', 'float': 'right'}),
    # dash_table.DataTable(data=trials_df[[i for i in trials_df.columns.to_list()]].to_dict('records'), page_size=10, style_table={'width': 'auto'})
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = trials_df[value]
    print(type(dff))
    fig = go.Figure(
        data=[go.Scatter(x=dff.index, y=dff.values, mode='markers')],
        layout=go.Layout(
            title=f"Trial Number vs. {value}",
            xaxis=dict(title="Trial Number"),
            yaxis=dict(title=f"{value}")
        )
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
