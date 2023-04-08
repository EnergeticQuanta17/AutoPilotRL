# Import packages
from dash import Dash, html, dash_table
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

for i in trials_df.columns.to_list():
    print(trials_df[i].head())
    print()

app = Dash(__name__)

print(trials_df[['number', 'value', 'duration']].to_dict('records'))

for i in ['datetime_start', 'datetime_complete', 'duration']:
    trials_df[i] = trials_df[i].astype(str)

app.layout = html.Div([
    html.Div(children='My First Study Data'),
    dash_table.DataTable(data=trials_df[[i for i in trials_df.columns.to_list()]].to_dict('records'), page_size=10, style_table={'width': 'auto'})
])

if __name__ == '__main__':
    app.run_server(debug=True,port=1026)
