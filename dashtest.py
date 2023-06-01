import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash
from dash import dcc
from dash import html
import dash_player as player
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import time

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

processing_modal = dbc.Modal(
    [
        dbc.ModalHeader("Processing Status"),
        dbc.ModalBody(
            dcc.Loading(
                id="loading",
                children=[html.Div(id="processing-status")]
            )
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-processing-modal", className="ml-auto")
        ),
    ],
    id="processing-modal",
)

app.layout = html.Div([
    html.H1("Main Page"),
    dbc.Button("Show Processing Status", id="show-processing-modal"),
    processing_modal
])


@app.callback(
    Output("processing-status", "children"),
    [Input("loading", "fullscreen")]
)
def update_processing_status(fullscreen):
    if fullscreen:
        return "Processing..."
    else:
        return "Done."


@app.callback(
    Output("processing-modal", "is_open"),
    [Input("show-processing-modal", "n_clicks"), Input("close-processing-modal", "n_clicks")],
    [State("processing-modal", "is_open")]
)
def toggle_processing_modal(n_show, n_close, is_open):
    if n_show or n_close:
        return not is_open
    return is_open


@app.callback(
    Output("loading", "fullscreen"),
    [Input("show-processing-modal", "n_clicks")]
)
def simulate_processing(n_clicks):
    if not n_clicks:
        return False

    for i in range(5):
        time.sleep(1)
    return False


if __name__ == "__main__":
    app.run_server(debug=True)
