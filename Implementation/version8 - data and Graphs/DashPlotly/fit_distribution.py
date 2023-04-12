import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Generate sample data
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

data = trials_df['value']
# Define the distributions to test
dist_names = ['expon', 'beta', 'rayleigh']
# dist_names = ['norm', 'expon', 'gamma', 'beta', 'rayleigh']

# Fit each distribution to the data and calculate its goodness of fit using Kolmogorov-Smirnov test
results = []
for dist_name in dist_names:
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    ks_test = stats.kstest(data, dist_name, params)
    results.append((dist_name, params, ks_test.pvalue))

# Sort the results by the goodness of fit p-value
results.sort(key=lambda x: x[2], reverse=True)

# Select the best fitted distribution and plot its probability density function
best_dist_name, best_params, best_pvalue = results[0]
best_dist = getattr(stats, best_dist_name)

print("The best distribition according to p-value is:", str(getattr(best_dist, 'name')))

pdf = best_dist.pdf(np.linspace(-5, 5, 1000), *best_params)
plt.plot(np.linspace(-5, 5, 1000), pdf, label=best_dist_name)
plt.hist(data, bins=30, density=True, alpha=0.5)
plt.legend()
plt.show()


