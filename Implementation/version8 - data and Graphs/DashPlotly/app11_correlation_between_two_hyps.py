from dash import Dash, html, dash_table, dcc, Input, Output, callback
import plotly.express as px

from .util import *

def correlation_hyp_vs_hyp(trials_df):
    trials_df = normalize_df(trials_df, normalize=False)

    
    corr = trials_df.corr()
    corr = significant_digits(corr)

    FULL_HTML = html.Div([
        html.Div(style={'clear': 'both'}),
        html.Br(), html.Br(), html.Br(),
        html.Hr(),

        html.H1('Correlation between Two Hyperparameters'),
        
        html.Div([
            html.Div([
                dcc.Dropdown(
                    trials_df.columns.to_list(),
                    'duration',
                    id='xaxis-column'
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    trials_df.columns.to_list(),
                    'reward',
                    id='yaxis-column'
                ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ]),

        dcc.Graph(id='indicator-graphic'),
        
        

        html.H1('Correlation Matrix'),
        dash_table.DataTable(
            id='correlation-table',
            columns=[{"name": i, "id": i} for i in corr.columns],
            data=corr.to_dict('records')
        )
    ])


    @callback(
        Output('indicator-graphic', 'figure'),
        Input('xaxis-column', 'value'),
        Input('yaxis-column', 'value'),
    )
    def update_graph(xaxis_column_name, yaxis_column_name):
        fig = px.scatter(trials_df, x=trials_df[xaxis_column_name],
                        y=trials_df[yaxis_column_name],
                        color='reward',
                        color_continuous_scale="bluered",
                        )

        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

        
        return fig
    
    return FULL_HTML