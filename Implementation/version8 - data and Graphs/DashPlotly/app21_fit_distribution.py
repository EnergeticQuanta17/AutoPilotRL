import winsound
import pyttsx3

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd

import optuna
import urllib

import scipy.stats as st

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

    for i in ['datetime_start', 'datetime_complete', 'state']:
        trials_df = trials_df.drop(i, axis=1)

    trials_df['duration'] = trials_df['duration'].dt.total_seconds()
    
    if(normalize):
        trials_df = trials_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return trials_df

trials_df = preprocess_df(study, True)


import plotly.graph_objs as go
import numpy as np


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
def update_graph(column_names):
    data = []
    
    for col in column_names:
        # fit the distributions to the data
        dist_results = []
        for dist_name in ['gamma', 'beta', 'rayleigh', 'norm', 'pareto', 'uniform']:
            dist = getattr(st, dist_name)
            param = dist.fit(trials_df[col].values)
            dist_results.append((dist_name, param))
        
        # select the best distribution
        best_dist_name, best_params = max(dist_results, key=lambda item: st.kstest(trials_df[col], item[0], item[1]).statistic)
        print(f"{col}: {best_dist_name} ({best_params})")
        
        # create the x and y values for the fitted distribution
        x_range = np.linspace(trials_df[col].min(), trials_df[col].max(), num=100)
        dist = getattr(st, best_dist_name)
        y_vals = dist.pdf(x_range, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        
        # create a scatter trace for the fitted distribution
        trace = go.Scatter(x=x_range, y=y_vals, name=col, opacity=0.7)
        data.append(trace)

    # set the layout
    layout = go.Layout(
        title="Best Fit Distribution Plot",
        xaxis=dict(title='Values'),
        yaxis=dict(title='Density'),
        height=700
    )
    
    # create the figure and return it
    fig = go.Figure(data=data, layout=layout)
    return fig

winsound.Beep(440, 500)
speak('Dash App Updated !')

if __name__ == "__main__":
    app.run_server(debug=True)