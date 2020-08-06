import dash_html_components as html
import dash_core_components as dcc


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])

def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("logo_minion.png"),
                        className="logo",
                    ),
                    html.A(
                        html.Button("Learn More", id="learn-more-button"),
                        href="http://github.com/AmauryLejay/auto_bi/",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("Auto-Generated BI report")],
                        className="seven columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/auto_bi/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    # TO DO - add session for new categories
    menu = html.Div(
        [
            dcc.Link(
                "Basic KPIs",
                href="/auto_bi/basic_kpis_only",
                className="tab first",
            ),
            dcc.Link(
                "Analysis",
                href="/auto_bi/analysis",
                className="tab",
            ),
            dcc.Link(
                "Full View",
                href="/auto_bi/full-view-analysis",
                className="tab",
            ),
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = [html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),]
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

def plotly_bar_chart(df_to_plot,target_value,feature_entropy_ranking,i):
    
    # Change i to assign a unique value to i 
    i += 1
    
    # Declare the fig
    fig = px.bar(df_to_plot, x='visu_axis', y=id_col, text=id_col,labels={'visu_axis':'Feature',id_col:'count'})
    fig.update_traces(textposition='outside')
    fig.update_layout(barmode='group',
                width=800, height=500,
                legend={'x': 0, 'y': 1},
                legend_title='<b> Target Value </b>',# For categorical only
                showlegend=True,
                xaxis_tickangle=-45,
                title={'text':" ".join([str(target_value), df_features_entropy.index[feature_entropy_ranking]]),
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
    
    # Store the fig in the dictionnary
    dic_fig[f'fig_{i}'] = fig
    
    return i