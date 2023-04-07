from operator import imod
from dash import Dash, dcc, html

app = Dash(__name__)

app.layout = html.Div([
    dcc.ConfirmDialog(
        id='confirm',
        message='Danger danger! Are you sure you want to continue?'
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)