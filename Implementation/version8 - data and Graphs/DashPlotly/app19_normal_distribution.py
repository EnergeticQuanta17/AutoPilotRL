from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

from .util import *

def normal_distribution_fit(trials_df):
    trials_df = normalize_df(trials_df, normalize=True)
    trials_df = change_column_names(trials_df)

    norm_data = []
    for col in trials_df.columns:
        norm_data.append(np.random.normal(trials_df[col].mean(), trials_df[col].std(), 1000))


    FULL_HTML = html.Div(
        [
            html.Div(style={'clear': 'both'}),
            html.Br(), html.Br(), html.Br(),
            html.Hr(),

            html.H1("Normal Distribution fit of Hyperparameter's Distributions", style={'padding': '0'}),

            html.P('Hyperparameters to be plotted:'),
            dcc.Checklist(
                id="x-axis19",
                options=trials_df.columns.to_list(),
                value=trials_df.columns.to_list(),
                inline=True,
            ),
            
            dcc.Graph(id="graph19"),
        ]
    )

    @callback(
        Output("graph19", "figure"),
        Input("x-axis19", "value"),
    )
    def update_graph(column_name):
        data = []
        for i, col in enumerate(column_name):
            x_range = np.linspace(norm_data[i].min(), norm_data[i].max(), num=100)
            y_vals = 1 / (trials_df[col].std() * np.sqrt(2 * np.pi)) * np.exp(- (x_range - trials_df[col].mean()) ** 2 / (2 * trials_df[col].std() ** 2))
            trace = go.Scatter(x=x_range, y=y_vals, name=col, opacity=0.7)
            data.append(trace)

        layout = go.Layout(
            title="Normal Distribution Plot",
            xaxis=dict(title='Values'),
            yaxis=dict(title='Count'),
            height=700
        )
        fig = go.Figure(data=data, layout=layout)
        return fig
    
    return FULL_HTML