from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px

from .util import *

def dim3_correlation(trials_df):
    trials_df = normalize_df(trials_df, normalize=False)
    trials_df = change_column_names(trials_df)

    FULL_HTML = html.Div(
        [
            html.Div(style={'clear': 'both'}),
            html.Br(), html.Br(), html.Br(),
            html.Hr(),

            html.H1("Trivariate Correlation"),
            html.Div([
            html.Div([
                dcc.Dropdown(
                    trials_df.columns.to_list(),
                    'duration',
                    id='xaxis-column17'
                ),
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    trials_df.columns.to_list(),
                    'reward',
                    id='yaxis-column17'
                ),
            ], style={'width': '33%', 'float': 'right', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    trials_df.columns.to_list(),
                    'reward',
                    id='zaxis-column17'
                ),
            ], style={'width': '33%', 'float': 'right', 'display': 'inline-block'})
        ]),
            dcc.Graph(id="graph17"),
        ]
    )


    @callback(
        Output("graph17", "figure"),
        Input('xaxis-column17', 'value'),
        Input('yaxis-column17', 'value'),
        Input('zaxis-column17', 'value')
    )
    def update_chart(xaxis_column_name, yaxis_column_name, zaxis_column_name):

        fig = px.scatter_3d(
            trials_df,
            x=xaxis_column_name,
            y=yaxis_column_name,
            z=zaxis_column_name,
            color="reward",
            hover_data=trials_df.columns.to_list(),
            color_continuous_scale="bluered",
        )
        fig.update_layout(width=1000, height=800)
        return fig
    
    return FULL_HTML