import plotly.express as px
df = px.data.iris()
fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")
fig.write_html('Scatter_plot_matrices_to_observe_correlation.html', auto_open=True)