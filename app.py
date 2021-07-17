import os
import pathlib
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
from data import *
from dash import callback_context



# Initialize app

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    prevent_initial_callbacks=True
)
app.title = "CI analytics system"
server = app.server

# Load data




DEFAULT_OPACITY = 0.8


# Get commit

def get_commits ():
    return get_last_five_commits()

# App layout

app.layout = html.Div(
    id="root",
    children=[

        html.Div(
            id="header",
            children=[
                #html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                html.H4(children="Analytics for Python Projects"),
                html.P(
                    id="description",
                    children="Dashboard for analysing data collected during Continuous Integration",
                ),
            ],
        ),
        html.Div(
              id="container",
              children=[
              html.Button(
                  id="btn-0",
                  className="btn",
                  n_clicks=0,
                  children=get_commits()[0],
                 
              ),
              html.Button(
                  id="btn-1",
                  className="btn",
                  n_clicks=0,
                  children=get_commits()[1]
              ),
              html.Button(
                  id="btn-2",
                  className="btn",
                  n_clicks=0,
                  children=get_commits()[2]
              ),
              html.Button(
                  id="btn-3",
                  className="btn",
                  n_clicks=0,
                  children=get_commits()[3]
              ),
              html.Button(
                  id="btn-4",
                  className="btn",
                  n_clicks=0,
                  children=get_commits()[4]
              )        

              ]
        ),

    ],
)




# bottom_click_style = {
#      "border-radius": "0",
#      "border-color": "transparent",
#      "border-bottom": "rgb(85, 255, 241) solid 0.25rem",
#      "border-bottom-right-radius": "1px" ,
#      "border-bottom-left-radius": "1px" ,
#      "color": "white"
# }

#@app.callback( Output("button1", 'style'), [Input("button1", "n_clicks")])
# @app.callback([Output("button"+str(i), 'style') for i in range(5)], [Input("button"+str(i), "n_clicks") for i in range(5)])
@app.callback(
    [Output(f"btn-{i}", "className") for i in range(0, 5)],
    [Input(f"btn-{i}", "n_clicks") for i in range(0, 5)],
)
def focus_button(*args):
    ctx = dash.callback_context

    if not ctx.triggered or not any(args):
        return ["btn" for _ in range(0, 5)]

    # get id of triggering button
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    id = button_id.split("-")[1]
    id_to_int = int(id)
    if get_commits()[id_to_int] == 'x':
        return ["btn" if get_commits()[id_to_int] == 'x' else "btn" for i in range(0, 5)]

    return [
        "btn active" if button_id == f"btn-{i}" else "btn" for i in range(0, 5)
    ]





if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port = 8050)
