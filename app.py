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
import numpy as np
from paper.analysis import *

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
def get_timeseries_data(commit_value):

    # Getting the commit value first   
    #commit_value = get_last_commit()
    
    # Get dataframe for resource topic based on last commit
    df = get_dataframe_unit_test_by_commit('resource', commit_value)
    # slice dataframe for test_name and execution time only
    test_df = get_test_resource_information(df, 'execution_time')
    return test_df

""""
Getting total execution time for each CI run
"""

def get_total_execution_time(commit):
    df = get_timeseries_data(commit)
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


""""
Get status of the test result
"""

def test_summary_table():

    table_rows = []

    table_header = [
        html.Thead(html.Tr([html.Th("Test Case"), html.Th("Status")]))
    ]


    df=get_test_status()
    df = df [['test_name', 'result']]
    for index, row in df.iterrows():
        if row['result'] == 1:
            indicator = daq.Indicator( value=True, color="#f55b5b")  
        else:
            indicator = daq.Indicator( value=True, color="#15b7e8")  
        row= html.Tr([html.Td(row['test_name']), html.Td(indicator)])
        table_rows.append(row)
    table_body = [html.Tbody(table_rows)]
    

    return dbc.Table(table_header + table_body, bordered=True) 



def get_current_commit_from_click_context(ctx):
    
    
   # If we dont click on a button but some other context is handled e.g., selection from drop down menu
    if 'value' in ctx.triggered[0]["prop_id"] :

        commit_value = get_commits()[4]
       
    # If we click on a button
    elif "n_clicks" in ctx.triggered[0]["prop_id"]:
    
            # get id of triggering button
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # # Getting the id of the button
            id = button_id.split("-")[1]

            # # Convert the id from string to integer and then check if we don't have any commit, in such case there shouldnt be focus
            id_to_int = int(id)
            
            commit_value = get_commits()[id_to_int]

    # Initial state, we want the last commit
    elif "." in ctx.triggered[0]["prop_id"]:
        commit_value = get_commits()[4]
        
    update_commit_memory(commit_value)
    return commit_value 

""""
Getting features information for correlation matrix
"""
def get_data_for_matrix(matrix_data_type, commit_value):

    if matrix_data_type == 'incorrelation':

        # Getting the commit value first    

        df = merge_dataframes(commit_value)
        df.drop("_key", axis=1, inplace=True)
    elif matrix_data_type == 'testcorrelation':
        df = merge_test_level_dataframes(commit_value)
        df.drop("test_name", axis=1, inplace=True)
    return df



current_commit_value = get_last_commit()

""""
Getting average resource 
"""
# Getting the commit value first    
commit_value = get_last_commit()

# Get dataframe for resource topic based on last commit
df = get_dataframe_unit_test_by_commit('resource', commit_value)

# Calculate average memory for each test case
mem_df = get_resource_average(df, 'memory')

input_df = merge_dataframes(commit_value)
input_df.drop("_key", axis=1, inplace=True)

test_df = merge_test_level_dataframes(commit_value)
test_df.drop("test_name", axis=1, inplace=True)

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
            html.H3(children="Test Summary"),

            html.Div(className='level2-container',
            children =[ 
                            
                            daq.LEDDisplay(
                                className='left-pass-test-panel',
                                value=get_test_status()['test_name'].count(),
                                size=70,
                                label={'label': 'Total Executed Tests', 'style': {'font-size': '25px', 'font-weight': 'bold'}},
                                color='rgb(85, 255, 241)',
                                backgroundColor="#252e3f",
                                theme = {
                                'dark': 'true',
                                'detail': '#007439',
                                'primary': '#00EA64',
                                'secondary': '#6E6E6E',
                                },

                            ),

                            daq.LEDDisplay(
                                        className='pass-test-panel',
                                        label={'label': 'Tests Passed', 'style': {'font-size': '25px', 'font-weight': 'bold' }},
                                        labelPosition="top",
                                        value=get_test_success(),
                                        size=70,
                                        color='white',
                                        backgroundColor="transparent",
                                        theme = {
                                            'dark': 'true',
                                            'detail': '#007439',
                                            'primary': '#00EA64',
                                            'secondary': '#6E6E6E',
                                        },
                                    ),
                            daq.LEDDisplay(
                                            className='fail-test-panel',
                                            labelPosition="top",
                                            value=(get_test_status()['test_name'].count()-get_test_success()),
                                            size=70,
                                            color='white',
                                            backgroundColor="transparent",
                                            label={'label': 'Tests Failed', 'style': {'font-size': '25px', 'font-weight': 'bold'}},
                                            theme = {
                                                'dark': 'true',
                                                'detail': '#007439',
                                                'primary': '#00EA64',
                                                'secondary': '#6E6E6E',
                                            },
                                    ) ,
                                    
                                
                                


                        

                          
                               
                            html.Div(className='test-summary', 
                                children=[test_summary_table()],

                            )


                        ],
                
        
        ),

        html.H3(children="Test Execution Summary"),
        html.Div(className='level1-container',
                                                     
                children = [ 
                    html.Div(className='div-for-dropdown',
                        children=[
                            dcc.Dropdown(id='testselector',
                                        options=[
                                        {'label': i, 'value': i} for i in df["test_name"].unique()
                                        ],
                                        multi=True,
                                        value = [df['test_name'].drop_duplicates()[0]],
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
                                        value=get_total_execution_time(get_last_commit()),
                                        label="Total execution time",
                                        showCurrentValue=True,
                                        units='Hours'
                                    ),
                            ],
                    )

    
    
                ],
                
        ),
        html.H3(children="Memory Utilization and Correlation"),
        html.Div(className='level5-container',
            children = [
                  dcc.Graph(
                                id="bar-container",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#252e3f",
                                        paper_bgcolor="#252e3f",
                                    )
                                ),
                            ),
                html.Div(
                            id="correlation-graph",
                            children=[
                                dcc.Dropdown( 
                                        id='correlation-selection',
                                        options=[
                                            {'label': 'Input feature correlation', 'value': 'incorrelation'},
                                            {'label': 'Test feature correlation', 'value': 'testcorrelation'},
                                            
                                        ],
                                        placeholder="Select a correlation",
                                        value='incorrelation'
                                    ),
                                dcc.Graph(id='heatmap-chart'),
                            ]
                           
                    ),
            ],

        ),
        html.H3(children="Test Input Clustering"),
        html.Div(className='level3-container',
                children= [
                    html.Div(
                            id="input-knob",
                            children=[
                                    dcc.Dropdown( 
                                        id='input-cluster-dropdown',
                                        options=[
                                            {'label': 'Kmeans', 'value': 'kmeans'},
                                            {'label': 'Agglomerative', 'value': 'agglomerative'},
                                            {'label': 'AffinityPropagation', 'value': 'affinitypropagation'},
                                            {'label': 'DBScan', 'value': 'dbscan'},
                                             {'label': 'GMM', 'value': 'gmm'},
                                        ],
                                        placeholder="Select a Clustering Algorithm",
                                        value = 'kmeans'
                                    ),
                                   
                                    dcc.Dropdown(
                                        id='feature-dropdown',
                                        options=[
                                            {'label': i, 'value': i} for i in ['size','FunCalls', 'TExeStmt', 'TNoItr', 'ExeCond', 'mem', 'time']
                                        ],
                                        multi=True,
                                        placeholder="Select features",
                                        value=['mem', 'time', 'ExeCond']
                                    ),
                                     dcc.Checklist(
                                        id="checklist-input",
                                        options=[
                                            {'label': '2D', 'value': '2D'},
                                            {'label': '3D', 'value': '3D'}
                                        ],
                                        value=['2D'],
                                        labelStyle={'display': 'inline-block'}
                                    ),
                                    daq.Knob(
                                        id='input-setting-knob',
                                        min=1,
                                        max=10,
                                        scale={"start":1, "labelInterval":1, "interval": 1}
                                        
                                    ),
                                    
                            ] 
                    ),
                    html.Div(
                            id="input-scatter-graph",
                            children=[
                                dcc.Graph(id='input-clustering-graph'),
                            ]  
                    ),
                    
                    
                    
                ]
        ),
        html.H3(children="Unit Test Clustering"),
        html.Div(className='level6-container',
                children= [
                    html.Div(
                            id="test-knob",
                            children=[
                                    dcc.Dropdown( 
                                        id='test-cluster-dropdown',
                                        options=[
                                            {'label': 'Kmeans', 'value': 'kmeans'},
                                            {'label': 'Agglomerative', 'value': 'agglomerative'},
                                            {'label': 'AffinityPropagation', 'value': 'affinitypropagation'},
                                            {'label': 'DBScan', 'value': 'dbscan'},
                                             {'label': 'GMM', 'value': 'gmm'},
                                        ],
                                        placeholder="Select a Clustering Algorithm",
                                        value='kmeans'
                                    ),
                                   
                                    dcc.Dropdown(
                                        id='test-feature-dropdown',
                                        options=[
                                            {'label': i, 'value': i} for i in ['test_func_calls', 'line_numbers', 'per_test_iterations', 'encode_per_test_cond', 'result','mem', 'time']
                                        ],
                                        multi=True,
                                        placeholder="Select features",
                                        value=['test_func_calls', 'line_numbers', 'per_test_iterations']
                                       
                                    ),
                                     dcc.Checklist(
                                        id="checklist-test",
                                        options=[
                                            {'label': '2D', 'value': '2D'},
                                            {'label': '3D', 'value': '3D'}
                                        ],
                                        labelStyle={'display': 'inline-block'},
                                        value=['2D']
                                    ),
                                    daq.Knob(
                                        id='test-setting-knob',
                                        min=1,
                                        max=10,
                                        scale={"start":1, "labelInterval":1, "interval": 1}
                                    ),
                                    
                            ],
                            ),
                        html.Div(
                                    id="test-scatter-graph",
                                        children=[
                                            dcc.Graph(id='test-clustering-graph')
                                        ] ,
                        ),
                    
                                    
                ]),
        
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
    Output("progress-gauge", "value"),
     [
    Input('btn-0', 'n_clicks'),
    Input('btn-1', 'n_clicks'),
    Input('btn-2', 'n_clicks'),
    Input('btn-3', 'n_clicks'),
    Input('btn-4', 'n_clicks'),]
)
def update_progress_gauge(btn_0_click, btn_1_click, btn_2_click, btn_3_click, btn_4_click):
    
    ctx = dash.callback_context


    if ctx.triggered[0]["prop_id"] != '.':
    
        # get id of triggering button
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        # # Getting the id of the button
        id = button_id.split("-")[1]

        # # Convert the id from string to integer and then check if we don't have any commit, in such case there shouldnt be focus
        id_to_int = int(id)
        
        commit_value = get_commits()[id_to_int]

    else:
        commit_value = get_commits()[4]

    
    return get_total_execution_time(commit_value)


@app.callback(
    Output("timeseries-container", "figure"),
    [Input('testselector', 'value'),
    Input('btn-0', 'n_clicks'),
    Input('btn-1', 'n_clicks'),
    Input('btn-2', 'n_clicks'),
    Input('btn-3', 'n_clicks'),
    Input('btn-4', 'n_clicks'),]
   
)
def timeseries_chart(selected_dropdown_value, *args): #btn_0_click, btn_1_click, btn_2_click, btn_3_click, btn_4_click):
    

    ctx = dash.callback_context

    commit_value = get_current_commit_from_click_context(ctx)

    # Get dataframe for resource topic based on last commit
    df = get_dataframe_unit_test_by_commit('resource', commit_value)

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
    [Input('btn-0', 'n_clicks'),
    Input('btn-1', 'n_clicks'),
    Input('btn-2', 'n_clicks'),
    Input('btn-3', 'n_clicks'),
    Input('btn-4', 'n_clicks')]
)
def bar_chart(*args):

    ctx = dash.callback_context

    commit_value = get_current_commit_from_click_context(ctx)

    # Get dataframe for resource topic based on last commit
    df = get_dataframe_unit_test_by_commit('resource', commit_value)

    # Calculate average memory for each test case
    mem_df = get_resource_average(df, 'memory')

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

@app.callback(
Output("heatmap-chart", "figure"), 
   [Input("correlation-selection", "value"),
    Input('btn-0', 'n_clicks'),
    Input('btn-1', 'n_clicks'),
    Input('btn-2', 'n_clicks'),
    Input('btn-3', 'n_clicks'),
    Input('btn-4', 'n_clicks')
   ]
)
def update_heat_map(selection, *args):

    ctx = dash.callback_context

    current_commit_value = get_current_commit_from_click_context(ctx)

    df = get_data_for_matrix(selection, current_commit_value)
    #df.drop("test_name", axis=1, inplace=True)
    df=normalized_data(df,selection)

    calculate_p_values_for_correlation(df, current_commit_value, selection)

    corr = df.corr()
    
    ## This is done for test result. Otherwise, transformation will result in Nan.
    corr = corr.replace(np.NaN, 0 )

    trace = go.Heatmap(z=corr.values,
                  x=corr.index.values,
                  y=corr.columns.values)
    data = [trace]
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
              ),
              }

    return figure

@app.callback(
Output("input-clustering-graph", "figure"), 
[Input("input-cluster-dropdown", "value"),
Input("feature-dropdown", "value"),
Input("checklist-input", "value"),
Input("input-setting-knob", "value"),
Input("correlation-selection", "value"),
Input('btn-0', 'n_clicks'),
Input('btn-1', 'n_clicks'),
Input('btn-2', 'n_clicks'),
Input('btn-3', 'n_clicks'),
Input('btn-4', 'n_clicks')
]

)
def input_clustering(clustering_type, feature_name, check_list_type, cluster_value, *args):

    ctx = dash.callback_context

    commit_value = get_current_commit_from_click_context(ctx)

    # # Get dataframe for resource topic based on last commit
    df = get_dataframe_unit_test_by_commit('resource', commit_value)

    # # Calculate average memory for each test case
    mem_df = get_resource_average(df, 'memory')

    input_df = merge_dataframes(commit_value)
    input_df.drop("_key", axis=1, inplace=True)


    if cluster_value == None:
        cluster_value = 4  

    if len (feature_name) < 2:
        feature_name = ["mem", "time"]

    df = normalized_data(input_df, 'incorrelation')

    cluster_write_to_file(df, None,commit_value)
    #elbow_identification(df, commit_value)
    #compare_silhouette(df, commit_value)
    cluster_value = int(cluster_value)
    df = clustering(clustering_type,df, cluster_value)


    if '2D' in check_list_type:
             fig = px.scatter(x= df[feature_name[0]],y= df[feature_name[1]], color=df['Cluster'])
    elif '3D' in check_list_type:
            fig= px.scatter_3d(df, x = feature_name[0], y=feature_name[1], z=feature_name[2],
              color='Cluster', opacity = 0.8, size='size', size_max=70, title="K-means")
    #fig = dict()
    fig.update_layout(colorway=['rgb(85, 255, 241)'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'t': 50},
                  height=250,
                  hovermode='x',
                  autosize=True,)
    return fig



@app.callback(
Output("test-clustering-graph", "figure"), 
[Input("test-cluster-dropdown", "value"),
Input("test-feature-dropdown", "value"),
Input("checklist-test", "value"),
Input("test-setting-knob", "value"),
Input('btn-0', 'n_clicks'),
Input('btn-1', 'n_clicks'),
Input('btn-2', 'n_clicks'),
Input('btn-3', 'n_clicks'),
Input('btn-4', 'n_clicks')
]

)
def test_clustering(clustering_type, feature_name, check_list_type, cluster_value, *args):
    
    ctx = dash.callback_context

    commit_value = get_current_commit_from_click_context(ctx)

    # # Get dataframe for resource topic based on last commit
    df = get_dataframe_unit_test_by_commit('resource', commit_value)

    # # Calculate average memory for each test case
    mem_df = get_resource_average(df, 'memory')

    test_df = merge_test_level_dataframes(commit_value)
    test_df.drop("test_name", axis=1, inplace=True)

    if cluster_value == None:
        cluster_value = 2  

    if len (feature_name) < 2:
        feature_name = ["mem", "time"]

    df = normalized_data(test_df, 'testcorrelation')
    cluster_value = int(cluster_value)
    df = clustering(clustering_type,df,cluster_value)

    if '2D' in check_list_type:
             fig = px.scatter(x= df[feature_name[0]],y= df[feature_name[1]], color=df['Cluster'])
    elif '3D' in check_list_type:
            fig= px.scatter_3d(df, x = feature_name[0], y=feature_name[1], z=feature_name[2],
              color='Cluster', opacity = 0.8, size_max=70, title="K-means")
    #fig = dict()
    fig.update_layout(colorway=['rgb(85, 255, 241)'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'t': 50},
                  height=250,
                  hovermode='x',
                  autosize=True,)
    return fig



if __name__ == "__main__":

    # We store commits before running the app. dc storage isn't working properly. This fulfills our requirement at the moment.
    # We will have to check the issue
    init_commit_memory(get_commits())
    app.run_server(debug=True, host='0.0.0.0', port = 8050)
