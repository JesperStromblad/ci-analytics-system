import os
import pathlib
import re
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
from data import *
from dash import callback_context
import plotly.graph_objects as go
import dash_daq as daq
import plotly.express as px


# Initialize app

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    #prevent_initial_callbacks=True,
)
app.title = "CI analytics system"
server = app.server




##---------------------------
## Functions for getting data
##---------------------------

""""
Getting data for unit test vs execution time
"""
def get_timeseries_data():

    # Getting the commit value first    
    commit_value = get_last_commit()

    
    # Get dataframe for resource topic based on last commit
    df = get_dataframe_unit_test_by_commit('resource', commit_value)
    # slice dataframe for test_name and execution time only
    test_df = get_test_resource_information(df, 'execution_time')
    return test_df

""""
Getting total execution time for each CI run
"""

def get_total_execution_time():
    df = get_timeseries_data()
    return (df['execution_time'].sum()) * 0.000277778


""""
Getting total number of tests with success and failure
"""
def get_test_status():

    # Getting the commit value first    
    commit_value = get_last_commit()

    return get_test_result_status('trace', commit_value)

def get_test_success():
    commit_value = get_last_commit()
    df = get_test_result_status('trace', commit_value)
    return df[df.result == 1].count().unique()[0]


def test_summary_table():
    table_header = [
                    html.Thead(html.Tr([html.Th("First Name"), html.Th("Last Name")]))
                   ]

    row1 = html.Tr([html.Td("Arthur"), html.Td(daq.Indicator(
                                                id='my-daq-indicator',
                                                value=True,
                                                color="#00cc96"
                                                )  )])
    row2 = html.Tr([html.Td("Ford"), html.Td("Prefect")])
    row3 = html.Tr([html.Td("Zaphod"), html.Td("Beeblebrox")])
    row4 = html.Tr([html.Td("Trillian"), html.Td("Astra")])

    table_body = [
                    html.Tbody([row1, row2, row3, row4])
                 ]

    return dbc.Table(
                # using the same table as in the above example
                table_header + table_body,
                bordered=True,
                dark=True,
                hover=True,
                responsive=True,
                striped=True,
            )

""""
Getting average resource 
"""
# Getting the commit value first    
commit_value = get_last_commit()

# Get dataframe for resource topic based on last commit
df = get_dataframe_unit_test_by_commit('resource', commit_value)

# Calculate average memory for each test case
mem_df = get_resource_average(df, 'memory')



## ----------------------------------------
## Initialization state for data collection
##-----------------------------------------




# App layout

app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.H4(children="Analytics for Python Projects"),
                html.P(
                    id="description",
                    children="Dashboard for analysing data collected during Continuous Integration",
                ),
            ],
        ),
        html.Div(id='temp'),
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
                  className="btn active",
                  n_clicks=0,
                  children=get_commits()[4]
              )        

              ]
        ),

        html.Div(className='level1-container',
                                                     
                children = [ 
                    html.Div(className='div-for-dropdown',
                        children=[
                            dcc.Dropdown(id='testselector',
                                        options=[
                                        {'label': i, 'value': i} for i in df["test_name"].unique()
                                        ],
                                        multi=True,
                                        value = df['test_name'].drop_duplicates()[0],
                                        style={'backgroundColor': '#1E1E1E'},
                                        className='stockselector'),

                            dcc.Graph(
                                id="timeseries-container",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#252e3f",
                                        paper_bgcolor="#252e3f",
                                    )
                                ),
                            ),


                        ],



                        style={'color': '#1E1E1E'}
                    ), 
    
                    html.Div(className='gauge-container',
                            children=[
                            daq.Gauge(
                                        color='rgb(85, 255, 241)',
                                        id="progress-gauge",
                                        max=24,
                                        min=0,
                                        value=get_total_execution_time(),
                                        label="Total execution time",
                                        showCurrentValue=True,
                                        units='Hours'
                                    ),
                            ],
                    )

    
    
                ],
                
        ),

        html.Div(className='level2-container',
            children =[ 
                            
                            daq.LEDDisplay(
                                className='left-pass-test-panel',
                                label="Total tests",
                                value=get_test_status()['test_name'].count(),
                                size=64,
                                color='rgb(85, 255, 241)',
                                backgroundColor="transparent"
                            ),
                            html.Div(className="success-panel",
                                children=[
                                       daq.LEDDisplay(
                                        className='pass-test-panel',
                                        label="Total tests passed",
                                        labelPosition="top",
                                        value=get_test_success(),
                                        size=20,
                                        color='#15b7e8',
                                        backgroundColor="transparent"
                                    ),
                                        daq.LEDDisplay(
                                            className='fail-test-panel',
                                            label="Total tests failure",
                                            labelPosition="bottom",
                                            value=(get_test_status()['test_name'].count()-get_test_success()),
                                            size=20,
                                            color='#f55b5b',
                                            backgroundColor="transparent"
                                    ) 
                                    
                                ]
                                


                            ),

                            dcc.Graph(
                                id="bar-container",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#252e3f",
                                        paper_bgcolor="#252e3f",
                                    )
                                ),
                            ),
                               
                            html.Div(className='test-summary', 
                                children=[test_summary_table()],

                            )


                        ],
                
        
        )
        
    ],
)

"""
Callback for button click and focus
"""

@app.callback(
    [Output(f"btn-{i}", "className") for i in range(0, 5)],
    [Input(f"btn-{i}", "n_clicks") for i in range(0, 5)]
)
def focus_button(*args):

    ctx = dash.callback_context

    if not ctx.triggered or not any(args):
        return ["btn" for _ in range(0, 5)]

    # get id of triggering button
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Getting the id of the button
    id = button_id.split("-")[1]

    

    # Convert the id from string to integer and then check if we don't have any commit, in such case there shouldnt be focus
    id_to_int = int(id)
    
    commit_value = get_commits()[id_to_int]

    if commit_value  == 'x':
        return ["btn" if get_commits()[id_to_int] == 'x' else "btn" for i in range(0, 5)] 

    update_commit_memory(commit_value)

    # If there is a click on the button, then focus on the button
    return [
        "btn active" if button_id == f"btn-{i}" else "btn" for i in range(0, 5)
    ]


@app.callback(
    Output("timeseries-container", "figure"),
    [Input('testselector', 'value')]
)
def timeseries_chart(selected_dropdown_value):
 
    trace = []
    df_sub = df
    # Draw and append traces for each stock
    for stock in selected_dropdown_value:
        trace.append(go.Scatter(x=df_sub[df_sub['test_name'] == stock].index,
                                 y=df_sub[df_sub['test_name'] == stock]['execution_time'],
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center')     
                                 )
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'t': 50},
                  height=250,
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Unit Test Execution', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'showticklabels': False, 'range': [df_sub.index.min(), df_sub.index.max()]},
                  yaxis_title="Execution time (s)",
                  xaxis_title="Test Case/s (input)",
              ),
              }

    return figure





# # Bar graph 
@app.callback(
    Output("bar-container", "figure"),
    [Input("temp", "children")],
)
def bar_chart(no_args):
    trace= go.Bar(x=mem_df.test_name,
                                 y=mem_df.memory,
                                 opacity=0.7,
                                 textposition='outside')     
                                 

    data= [trace]
    # Define Figure
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=['rgb(85, 255, 241)'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'t': 50},
                  height=250,
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Memory utilization', 'font': {'color': 'white'}, 'x': 0.5},
                  yaxis= {'type': 'log'},
                  xaxis={'showticklabels': False, 'range': [mem_df.index.min(), mem_df.index.max()]},
                  yaxis_title="Avg mem",
                  xaxis_title="Test Cases",
              ),
              }

    return figure

if __name__ == "__main__":

    # We store commits before running the app. dc storage isn't working properly. This fulfills our requirement at the moment.
    # We will have to check the issue
    init_commit_memory(get_commits())
    app.run_server(debug=True, host='0.0.0.0', port = 8050)
