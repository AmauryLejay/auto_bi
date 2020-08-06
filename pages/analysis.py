import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from utils import Header, make_dash_table, plotly_bar_chart
import pandas as pd
import pathlib
import auto_bi
import numpy as np

def add_go_bar(x,y,name):
                """
                create a dash bar chart
                input : 
                x - list values
                y - list values
                name - string of bar chart name
                
                output: 
                go.bar object
                """
                return go.Bar(
                                                    x=x,
                                                    y=y,
                                                    marker={
                                                        "color": "#1f2e4f",
                                                        "line": {
                                                            "color": "rgb(255, 255, 255)",
                                                            "width": 2,
                                                        },
                                                    },
                                                    name=name,
                                                ),

def html_div_plots(df_to_plot,feature,i):
    return html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        str(feature),
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id=f"graph-{i}",
                                        figure={
                                            "data": add_go_bar(list(df_to_plot["visu_axis"]),list(df_to_plot['count']),feature),
                                            "layout": go.Layout(
                                                autosize=False,
                                                bargap=0.35,
                                                font={"family": "Raleway", "size": 10},
                                                height=200,
                                                hovermode="closest",
                                                legend={
                                                    #"x": -0.0228945952895,
                                                    #"y": -0.189563896463,
                                                    "x": 0.0228945952895,
                                                    "y": 1.05,
                                                    "orientation": "h",
                                                    "yanchor": "top",
                                                },
                                                margin={
                                                    "r": 25,
                                                    "t": 25,
                                                    "b": 65,
                                                    "l": 25,
                                                },
                                                showlegend=True,
                                                title="",
                                                width=330,
                                                xaxis={
                                                    "autorange": True,
                                                    "range": [-0.5, 4.5],
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "category",
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    "range": [0, 22.9789473684],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),


def create_layout(app,dic_groupby_df,df_feature_entropy,entropy_threshold,target,headers = True):
    
    df_feature_entropy = df_feature_entropy[df_feature_entropy['entropy']>entropy_threshold]
    i = 0
    list_html_div_plots = []
    row_list = []

    for feature in df_feature_entropy.index:
        df_visu = dic_groupby_df[feature]
        df_visu = df_visu.sort_values(by = 'count').reset_index()
        df_visu['visu_axis'] = df_visu.drop(['count',target], axis=1).astype(str).apply('_'.join, axis=1)

        for target_value in df_visu[target].unique():
            df_to_plot = df_visu[df_visu[target]==target_value]
            list_html_div_plots.append(html_div_plots(df_to_plot,feature,i))
            i += 1

    if headers: 
        return html.Div(
            [
                html.Div([Header(app)]),
                # page 1
                html.Div(
                    # Row X 
                    [div[0] for div in list_html_div_plots],
                    className="sub_page",
                ),
            ],
            className="page",
        )

    else :
        return html.Div(
            [
                # page 1
                html.Div(
                    # Row X 
                    [div[0] for div in list_html_div_plots],
                    className="sub_page",
                ),
            ],
            className="page",
        )