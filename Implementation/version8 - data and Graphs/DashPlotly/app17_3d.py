import winsound
import pyttsx3

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd

import optuna
import urllib

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-30) 

def speak(text):
    engine.say(text)
    engine.runAndWait()

def study_loader():
    study_name = "study-26"
    url = "https://raw.githubusercontent.com/SBhat2615/AutoPilotRL/main/Implementation/version8%20-%20data%20and%20Graphs/study-26.db"
    storage_name = f"sqlite:///{study_name}.db"

    with urllib.request.urlopen(url) as response:
        with open(study_name + ".db", 'wb') as f:
            f.write(response.read())

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )

    return study

study = study_loader()

def preprocess_df(study, normalize=True):
    trials_df = study.trials_dataframe()

    for i in ['datetime_start', 'datetime_complete']:
        trials_df[i] = trials_df[i].apply(lambda x: int(x.timestamp()))

    trials_df['duration'] = trials_df['duration'].dt.total_seconds()
    trials_df = trials_df.drop('state', axis=1)

    if(normalize):
        trials_df = trials_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return trials_df

trials_df = preprocess_df(study, False)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("Observing correlation between 3 hyperparameters"),
        html.Div([
        html.Div([
            dcc.Dropdown(
                trials_df.columns.to_list(),
                'duration',
                id='xaxis-column'
            ),
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                trials_df.columns.to_list(),
                'value',
                id='yaxis-column'
            ),
        ], style={'width': '33%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                trials_df.columns.to_list(),
                'value',
                id='zaxis-column'
            ),
        ], style={'width': '33%', 'float': 'right', 'display': 'inline-block'})
    ]),
        dcc.Graph(id="graph"),
    ]
)


@app.callback(
    Output("graph", "figure"),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('zaxis-column', 'value')
)
def update_chart(xaxis_column_name, yaxis_column_name, zaxis_column_name):

    fig = px.scatter_3d(
        trials_df,
        x=xaxis_column_name,
        y=yaxis_column_name,
        z=zaxis_column_name,
        color="value",
        hover_data=trials_df.columns.to_list(),
        color_continuous_scale="bluered",
    )
    fig.update_layout(width=1400, height=1000)
    return fig

winsound.Beep(440, 500)
speak('Dash App Updated !')

if __name__ == "__main__":
    app.run_server(debug=True)