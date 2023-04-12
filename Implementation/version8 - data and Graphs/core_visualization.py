import winsound
import pyttsx3

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-30) 

def speak(text):
    engine.say(text)
    engine.runAndWait()

from dash import Dash, html, dcc
from DashPlotly import *

import optuna
import pyttsx3
import urllib

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-30) 

def speak(text):
    engine.say(text)
    engine.runAndWait()

class Visualizer:
    def __init__(self, study_name, url=""):
        self.study_name = study_name
        # self.url = url

        self.study = self.study_loader()

        self.trials_df = self.study.trials_dataframe()
        self.trials_df = self.trials_df.rename(columns={'value': 'reward'})

        # self.trials_df = self.preprocess_df(normalize=True)

        # self.graph_generator(self.trials_df)


    def study_loader(self):
        # storage_name may be a problem
        # url need to updated to relative path

        url = "https://raw.githubusercontent.com/SBhat2615/AutoPilotRL/main/Implementation/version8%20-%20data%20and%20Graphs/study-26.db"
        storage_name = f"sqlite:///{self.study_name}.db"

        with urllib.request.urlopen(url) as response:
            with open(self.study_name + ".db", 'wb') as f:
                f.write(response.read())

        study = optuna.load_study(
            study_name=self.study_name,
            storage=storage_name
        )

        return study

    def preprocess_df(self, normalize=True):
        trials_df = self.study.trials_dataframe()

        for i in ['datetime_start', 'datetime_complete', 'state']:
            trials_df = trials_df.drop(i, axis=1)

        trials_df['duration'] = trials_df['duration'].apply(lambda x: x.total_seconds())
        
        if(normalize):
            trials_df = trials_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        return trials_df

    def graph_generator(self):
        table1 = StudyTable(self.trials_df)
        table2 = ScatterLine(self.trials_df)
        table3 = ColorScaleHistogram(self.trials_df)
        table4 = InterCorrelation(self.trials_df)
        table5 = TrivariateCorrelation(self.trials_df)

        all_elements = [
            table1,
            table2,
            table3,
            table4,
            table5,
        ]

        table_divs = [html.Div([table]) for table in all_elements]

        app.layout = html.Div(table_divs)

v = Visualizer('study-26')
# trials_df = v.preprocess_df()

app = Dash(__name__)

v.graph_generator()

winsound.Beep(440, 500)
speak('Dash App Updated !')

if __name__ == "__main__":
    app.run_server(debug=True, port=1026)