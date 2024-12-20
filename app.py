from dash import Dash, dcc, html, Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
import json

import geopandas as gpd
import pandas as pd

from app_helper import style_handle, style, hover_style
import numpy as np
import plotly.express as px

pd.options.display.max_columns = 150

# Load the data
heb_dict_df = pd.read_csv('data/heb_english_dict.csv', index_col=0)
stats_data_gdf = gpd.read_file('data/stat_pop_simpl_votes_2022.geojson')

# Setting up dictionaries and classes for the map
col_rename = heb_dict_df.T.set_index(0)[1].to_dict()
colors_dict = heb_dict_df.T.set_index(0).loc['party_1':][2].to_dict()
colors_dict2 = heb_dict_df.T.set_index(1).loc['labor':][2].to_dict()
classes = list(colors_dict.keys())
colorscale = list(colors_dict.values())

# prepare and spatial data and covnert column names
stats_data_gdf.rename(columns=col_rename, inplace=True)
stats_data_gdf['sta_22_names'] = stats_data_gdf['sta_22_names'].str.replace(
    'No Name', '')
stats_data_gdf.to_crs('EPSG:4326', inplace=True)
stats_data = stats_data_gdf.__geo_interface__


# Create info control.
def get_info(feature=None, col_rename=col_rename):
    """
    Generate information about a given feature.

    Parameters
    ----------
    feature : dict, optional
        A dictionary representing a feature with properties. Default is None.
    col_rename : dict
        A dictionary used to rename columns based on feature properties.

    Returns
    -------
    list
        A list containing HTML elements with information about the feature.
    """
    header = []
    if feature is None:
        return

    return header + [html.B(feature["properties"]["Shem_Yishuv"]), html.B(" "), html.B(feature["properties"]["sta_22_names"]), html.Br(),
                     html.Span(col_rename.get(feature["properties"]["max_label"]))]


info = html.Div(children=get_info(), id="info", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"})


def generate_random_barplot(feature=None):
    feature_id = np.random.choice(stats_data_gdf['YISHUV_STAT11'].values)
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]

    # Select statistical area and select only the relevant columns, sort by votes
    selected_row = stats_data_gdf[stats_data_gdf['YISHUV_STAT11']
                                  == feature_id].iloc[0]['labor':]
    selected_row.drop(['geometry', 'max_label'], inplace=True)
    selected_row.sort_values(ascending=False, inplace=True)

    # Convert to percent
    percent = selected_row.values/selected_row.values.sum()

    # Prepare barplot
    top_ten = pd.Series(percent[0:6], index=selected_row.index[0:6])
    top_ten = top_ten[top_ten.values > 0]
    # Generate random data
    categories = top_ten.index
    values = top_ten.values
    # Create a bar plot
    fig = px.bar(x=categories, y=values, labels={
        'x': '', 'y': ''}, color=categories, color_discrete_map=colors_dict2)
    fig.update_layout(
        xaxis_tickangle=-90,
        yaxis=dict(range=[0, top_ten.max() if top_ten.max()
                   > 0.5 else 0.5], visible=False),
        height=800,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


app = Dash(title="Similar to me")
app.layout = html.Div([
    html.Div(
        [
            html.Div([dcc.Graph(id='elections_barplot')], style={
                     'width': '20%', 'display': 'inline-block', 'margin-right': '2%'}),
            html.Div([
                dl.Map([
                    dl.TileLayer(
                        url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                    dl.GeoJSON(id='stats_layer', data=stats_data,
                               hoverStyle=hover_style,
                               style=style_handle,
                               hideout=dict(
                                   colorscale=colorscale, classes=classes, style=style, hoverStyle=hover_style, colorProp="max_label")
                               ),
                    info
                ],
                    center=[32, 34.9],
                    zoom=12,
                    style={'height': '75vh'},
                    id='env_map',
                    dragging=True,
                    zoomControl=True,
                    scrollWheelZoom=True,
                    doubleClickZoom=True,
                    boxZoom=True,
                )

            ], style={'width': '74%', 'display': 'inline-block', 'margin-left': '2%'}
            )
        ],
    ),

])


@ app.callback(Output("info", "children"), Input("stats_layer", "hoverData"))
def info_hover(feature):
    return get_info(feature)


@ app.callback(Output('elections_barplot', 'figure'), Input('stats_layer', 'clickData'))
def update_barplot(clickData):
    return generate_random_barplot(clickData)


if __name__ == '__main__':
    app.run_server()
