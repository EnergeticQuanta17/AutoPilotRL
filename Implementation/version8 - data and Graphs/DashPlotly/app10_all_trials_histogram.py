from dash import Dash, html, dash_table, dcc, Input, Output, callback
import plotly.express as px

from .util import *

def color_scale_histogram(trials_df):
    trials_df = normalize_df(trials_df, normalize=False)
    trials_df = change_column_names(trials_df)
    FULL_HTML = html.Div(children=[
        html.Div(style={'clear': 'both'}),
        html.Br(), html.Br(), html.Br(),
        html.Hr(),

        html.H1('Color-Scale Hyperparameter Reward Correlation Visualization'),

        html.Label('Select column:'),
        dcc.Dropdown(options=[{'label': col, 'value': col} for col in trials_df.columns], value=trials_df.columns[0], id='dropdown-selection1'),

        dcc.Graph(id='graph-content1')
    ])

    @callback(
        Output('graph-content1', 'figure'),
        Input('dropdown-selection1', 'value')
    )
    def update_graph(value):
        fig = px.bar(trials_df, x="number", y=value, color="reward", barmode="group",  color_continuous_scale="bluered",)
        return fig
    
    return FULL_HTML