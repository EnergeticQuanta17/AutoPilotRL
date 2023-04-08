import plotly.express as px
import webbrowser

df = px.data.tips()
fig = px.scatter(
    df, x='total_bill', y='tip', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
)

fig.write_html('ordinary_least_squares.html', auto_open=True)