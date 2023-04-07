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

trials_df = preprocess_df(study, True)



import plotly.graph_objs as go
import numpy as np

# generate normal distribution data for each column in the dataframe
norm_data = []
for col in trials_df.columns:
    norm_data.append(np.random.normal(trials_df[col].mean(), trials_df[col].std(), 1000))


app = Dash(__name__)

app.layout = html.Div(
    [
        html.P('Hyperparameters to be plotted:'),
        dcc.Checklist(
            id="x-axis",
            options=trials_df.columns.to_list(),
            value=trials_df.columns.to_list(),
            inline=True,
        ),
        html.H1("Normal Distribution of chosen Hyperparameter Configuraitons", style={'padding': '0'}),
        dcc.Graph(id="graph"),
    ]
)

@app.callback(
    Output("graph", "figure"),
    Input("x-axis", "value"),
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

winsound.Beep(440, 500)
speak('Dash App Updated !')

if __name__ == "__main__":
    app.run_server(debug=True)