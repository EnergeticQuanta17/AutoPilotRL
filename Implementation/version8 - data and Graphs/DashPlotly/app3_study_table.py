from dash import Dash, html, dash_table, dcc, Input, Output, callback
import pandas as pd

from .util import *

def return_table_according_to_dataframe(df):
    print(df)
    table = dash_table.DataTable(data=df[[i for i in df.columns.to_list()]].to_dict('records'), 
    page_size=10,
    sort_action='native',
    style_mode='multi',
    )

    return table

def study_table(trials_df):
    print(trials_df)
    recorded_trials_df = trials_df

    trials_df = normalize_df(trials_df)    
    trials_df = significant_digits(trials_df)
    trials_df = change_column_names(trials_df)
    trials_df = reorder_cols_signif(trials_df)
  
    recorded_trials_df = normalize_df(recorded_trials_df, normalize=False)
    recorded_trials_df = significant_digits(recorded_trials_df)
    recorded_trials_df = change_column_names(recorded_trials_df)
    recorded_trials_df = reorder_cols_signif(recorded_trials_df)

    FULL_HTML = html.Div([
        html.H1('Explored Hyperparameter Configurations'),
        
       dcc.Checklist(
            id='checklist',
            options=[
                {'label': 'Normalized Data', 'value': 'df1'},
            ],
            value=['df1'],
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),

        dash_table.DataTable(
            id='data-table',
            columns=[{'name': col, 'id': col} for col in trials_df.columns],
            data=trials_df.to_dict('records')
        )
    ])

    @callback(
        Output('data-table', 'data'),
        Input('checklist', 'value')
    )
    def update_table(value):
        if 'df1' in value:
            return trials_df.to_dict('records')
        else:
            return recorded_trials_df.to_dict('records')
    
    return FULL_HTML