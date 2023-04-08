import winsound
import pyttsx3

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd

import optuna

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-30) 

def speak(text):
    engine.say(text)
    engine.runAndWait()

import urllib

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

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Comparing effectiveness of trials'),
    html.Label('Select column:'),
    dcc.Dropdown(options=[{'label': col, 'value': col} for col in trials_df.columns], value=trials_df.columns[0], id='dropdown-selection'),
    dcc.Graph(id='graph-content'),
    html.H2('The above graph can be used to see relationship between sampled Hyperpamameter Values and Reward Score'),
    html.Br(), html.Br(), html.Br(), html.Br(), 
    dash_table.DataTable(data=trials_df.to_dict('records'), page_size=5, style_table={'width': 'auto'})
])

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df[value], mode='markers', name=f"{value}"))
    fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df['value'], mode='markers', name='reward', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df[value], mode='lines', name=f'Line {value}'))
    fig.add_trace(go.Scatter(x=trials_df.index, y=trials_df['value'], mode='lines', name='Reward Line', marker=dict(color='red')))
    fig.update_layout(title=f'Comparing {value} and reward', xaxis=dict(title='Trial Number'), yaxis=dict(title=value))

    fig.update_layout(xaxis_rangeslider_visible=True)

    return fig

winsound.Beep(440, 500)
speak('Dash App Updated !')

if __name__ == '__main__':
    app.run_server(debug=True)



