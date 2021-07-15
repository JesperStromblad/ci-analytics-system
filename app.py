import os
import pathlib
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State


# Initialize app

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "CI analytics system"
server = app.server

# Load data



YEARS = [2003, 2004 , 2005 ,2006,2007]

BINS = [
    "0-2",
    "2.1-4",
    "4.1-6",
    "6.1-8",
    "8.1-10",
    "10.1-12",
    "12.1-14",
    "14.1-16",
    "16.1-18",
    "18.1-20",
    "20.1-22",
    "22.1-24",
    "24.1-26",
    "26.1-28",
    "28.1-30",
    ">30",
]

DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#bbffeb",
    "#98ffe0",
    "#79ffd6",
    "#6df0c8",
    "#69e7c0",
    "#59dab2",
    "#45d0a5",
    "#31c194",
    "#2bb489",
    "#25a27b",
    "#1e906d",
    "#188463",
    "#157658",
    "#11684d",
    "#10523e",
]

DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"

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
                            id="commit-slider-container",
                            children=[
                                html.P(
                                    id="commit-slider-text",
                                    children="Drag the slider when there are more than one commits",
                                ),
                                dcc.Slider(
                                    id="commit-years-slider",
                                    min=min(YEARS),
                                    max=max(YEARS),
                                    step=1,
                                    value=max(YEARS),
                                    marks={
                                        str(year): {
                                            "label": str(year),
                                            "style": {"color": "#06C6F6"},
                                        }
                                        for year in YEARS
                                    },
                                ),
                            ],
                        ),
   
    ],
)




if __name__ == "__main__":
    app.run_server(debug=True)
