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

for i in ['datetime_start', 'datetime_complete']:
    trials_df[i] = trials_df[i].apply(lambda x: int(x.timestamp()))

trials_df['duration'] = trials_df['duration'].dt.total_seconds()
trials_df = trials_df.drop('state', axis=1)

# Scale all columns to range of 0 to 1
trials_df = trials_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))


app = Dash(__name__)

app.layout = html.Div([
    html.H1('Comparing effectiveness of trials'),
    html.Label('Select column:'),
    dcc.Dropdown(options=[{'label': col, 'value': col} for col in trials_df.columns], value=trials_df.columns[0], id='dropdown-selection'),
    dcc.Graph(id='graph-content'),
    html.H2('The above graph can be used to see relationship between sampled Hyperpamameter Values and Reward Score'),
    html.Br(), html.Br(), html.Br(), html.Br(), 
    dash_table.DataTable(data=trials_df.to_dict('records'), page_size=10, style_table={'width': 'auto'})
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
    return fig

winsound.Beep(440, 500)
speak('Dash App Updated !')

if __name__ == '__main__':
    app.run_server(debug=True)
