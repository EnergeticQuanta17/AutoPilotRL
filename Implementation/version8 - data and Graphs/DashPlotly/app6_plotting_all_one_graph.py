
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go

from .util import *

def scatter_lines_dropdown(trials_df):
    trials_df = normalize_df(trials_df)
    FULL_HTML = html.Div([
        html.Div(style={'clear': 'both'}),
        html.Br(), html.Br(), html.Br(),
        html.Hr(),
        
        html.H1('Line-Plot Hyperparameter Reward Correlation'),

        html.Div([
            html.Label('Select column:'),
            dcc.Dropdown(options=[{'label': col, 'value': col} for col in trials_df.columns], value=trials_df.columns[0], id='dropdown-selection'),
        ], style={'width': '20%', 'float': 'left', 'margin-top': '80px'}),

        html.Div([
            dcc.Graph(id='graph-content', style={'height': '700px'}),
        ], style={'width': '80%', 'float': 'right'}),
    ])

    @callback(
        Output('graph-content', 'figure'),
        Input('dropdown-selection', 'value')
    )
    def update_graph(value):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df[value], mode='markers', name=f"{value}"))
        fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df['reward'], mode='markers', name='reward', marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df[value], mode='lines', name=f'Line {value}'))
        fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df['reward'], mode='lines', name='Reward Line', marker=dict(color='red')))
        fig.update_layout(title=f'Comparing {value} and reward', xaxis=dict(title='Trial Number'), yaxis=dict(title=value))
        return fig
    
    return FULL_HTML