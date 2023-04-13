from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

from scipy.stats import beta

from .util import *

def beta_distribution(trials_df):
    trials_df = normalize_df(trials_df, normalize=True)
    trials_df = change_column_names(trials_df)

    beta_params = {}
    for col in trials_df.columns.unique():
        a, b, loc, scale = beta.fit(trials_df[col])
        beta_params[col] = {'a': a, 'b': b, 'loc': loc, 'scale': scale}


    FULL_HTML = html.Div([
        html.Div(style={'clear': 'both'}),
        html.Br(), html.Br(), html.Br(),
        html.Hr(),
        html.H1("Beta Distribution fit of Hyperparameter's Distributions", style={'padding': '0'}),
        html.P('Hyperparameters to be plotted:'),
        dcc.Checklist(
            id="x-axis22",
            options=[{'label': col, 'value': col} for col in trials_df.columns.unique()],
            value=trials_df.columns.unique(),
            inline=True,
        ),
        dcc.Graph(id="graph22"),
    ])

    @callback(
        Output("graph22", "figure"),
        Input("x-axis22", "value"),
    )
    def update_graph(column_name):
        data = []
        for col in column_name:
            a = beta_params[col]['a']
            b = beta_params[col]['b']
            loc = beta_params[col]['loc']
            scale = beta_params[col]['scale']
            x_range = np.linspace(beta.ppf(0.001, a, b, loc=loc, scale=scale), beta.ppf(0.999, a, b, loc=loc, scale=scale), num=100)
            y_vals = beta.pdf(x_range, a, b, loc=loc, scale=scale)
            trace = go.Scatter(x=x_range, y=y_vals, name=col, opacity=0.7)
            data.append(trace)

        layout = go.Layout(
            title="Beta Distribution Plot",
            xaxis=dict(title='Values'),
            yaxis=dict(title='Density'),
            height=700
        )
        fig = go.Figure(data=data, layout=layout)
        return fig
    
    return FULL_HTML