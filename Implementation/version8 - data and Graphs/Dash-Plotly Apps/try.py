import numpy as np
import pandas as pd
import scipy.stats as st
import plotly.graph_objs as go

# generate sample data
data = np.random.normal(loc=10, scale=3, size=1000)

# define list of distributions to test
dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

# fit each distribution to the data and calculate the log-likelihood
results = []
for dist_name in dist_names:
    dist = getattr(st, dist_name)
    param = dist.fit(data)
    log_likelihood = np.sum(dist.logpdf(data, *param))
    results.append((dist_name, param, log_likelihood))

# sort the results by log-likelihood in descending order
results.sort(key=lambda x: -x[2])

# get the best distribution
best_dist_name, best_param, best_log_likelihood = results[0]

# plot the data and the fitted distribution
x = np.linspace(np.min(data), np.max(data), 100)
y = getattr(st, best_dist_name).pdf(x, *best_param)

fig = go.Figure()
fig.add_trace(go.Histogram(x=data, nbinsx=30))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=best_dist_name))
fig.show()
