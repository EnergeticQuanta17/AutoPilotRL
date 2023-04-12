from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px

from .util import *

def boxplot_hyperparameters(trials_df):
    trials_df = normalize_df(trials_df, normalize=True)
    trials_df = change_column_names(trials_df)

    FULL_HTML = html.Div(
        [
            html.Div(style={'clear': 'both'}),
            html.Br(), html.Br(), html.Br(),
            html.Hr(),

            html.H1("Box Plots of the Hyperparameter's Distributions"),

            html.P("x-axis:"),
            dcc.Checklist(
                id="x-axis",
                options=trials_df.columns.to_list(),
                value=trials_df.columns.to_list(),
                inline=True,
                labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                style={'display': 'grid', 'grid-template-columns': 'repeat(3, 1fr)'}
            ),
            dcc.Graph(id="graph"),
        ]
    )

    @callback(
        Output("graph", "figure"),
        Input("x-axis", "value"),
    )
    def update_graph(column_name):
        fig = px.box(trials_df, y=column_name)

        fig.update_xaxes(title_text="Hyperparameters")
        fig.update_yaxes(title_text="Distribution")
        return fig
    
    return FULL_HTML