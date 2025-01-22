from dash import Dash, dcc, html, Input, Output, State
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
from scipy.spatial import KDTree
import geopandas as gpd
import pandas as pd
from app_helper import won_style_handle, style, hover_style, map_analysis_radio_options, kde_classes, kde_colorscale, kde_style_handle
import numpy as np
import plotly.express as px
import json

pd.options.display.max_columns = 150


# Load the data
heb_dict_df = pd.read_csv('data/heb_english_dict.csv', index_col=0)
stats_data_original_gdf = gpd.read_file('data/stat_pop_simpl_votes_2022.geojson')

# Setting up dictionaries and classes for the map
col_rename = heb_dict_df.T.set_index(0)[1].to_dict()
colors_dict = heb_dict_df.T.set_index(0).loc['party_1':][2].to_dict()
colors_dict2 = heb_dict_df.T.set_index(1).loc['labor':][2].to_dict()
classes = list(colors_dict.keys())
colorscale = list(colors_dict.values())

# prepare and spatial data and covnert column names
stats_data_original_gdf.rename(columns=col_rename, inplace=True)
stats_data_original_gdf['sta_22_names'] = stats_data_original_gdf['sta_22_names'].str.replace(
    'No Name', '')
stats_data_original_gdf.to_crs('EPSG:4326', inplace=True)


def get_kdtree(stat_filter=stats_data_original_gdf.sample(1)['YISHUV_STAT11'].values[0], gdf=stats_data_original_gdf.copy()):

    kdf_filter_row = gdf[gdf['YISHUV_STAT11'] == stat_filter].iloc[0]
    kde_df = gdf.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
                       'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'], axis=1).copy()
    kdf_filter_row = kdf_filter_row.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
                                          'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'])
    # 3. Normalize the data
    kde_df = kde_df.apply(
        lambda p: p/p['bzb'], axis=1).drop('bzb', axis=1)
    kdf_filter_row = (kdf_filter_row/kdf_filter_row['bzb']).drop('bzb')

    # random_stat = kde_df.sample(1).values
    # Build the KDTree
    tree = KDTree(kde_df.values)

    # Query KDTree for the 3 nearest neighbors (including the row itself)
    distances, indices = tree.query(kdf_filter_row.values, k=len(kde_df))

    # Retrieve nearest rows using the indices
    gdf_kde = gdf.iloc[indices].copy()
    gdf_kde['kde_distnace'] = distances
    return gdf_kde

stats_data_gdf = gpd.GeoDataFrame()
stats_data_gdf = get_kdtree()
stats_data = stats_data_gdf.__geo_interface__

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


def generate_barplot(feature=None):
    if feature is not None:
        print(feature["properties"]["YISHUV_STAT11"])
        feature_id = feature["properties"]["YISHUV_STAT11"]
    else:
        feature_id = np.random.choice(stats_data_gdf['YISHUV_STAT11'].values)
        print('None')
    # Select statistical area and select only the relevant columns, sort by votes
    selected_row = stats_data_gdf[stats_data_gdf['YISHUV_STAT11']
                                  == feature_id].iloc[0]
    # Get title name (city + stat name)
    stat_name = f"{selected_row['Shem_Yishuv']} {selected_row['sta_22_names']}"[
        0:35]
    selected_row = selected_row['labor':]
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
        'x': '', 'y': ''}, color=categories, color_discrete_map=colors_dict2, title=stat_name)
    fig.update_layout(
        xaxis_tickangle=-90,
        yaxis=dict(range=[0, top_ten.max()+0.1 if top_ten.max()
                   > 0.5 else 0.5], visible=False),
        height=800,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(title_y=0.9, title_x=0.95)
    return fig


app = Dash(title="Similar to me")
app.css.append_css({
    'external_url': 'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'
})
app.layout = html.Div(children=[
    html.Div(
        [
            html.Div([
                html.H4("Map Analysis Options"),
                dcc.RadioItems(
                    id='raio_map_analysis',
                    options=map_analysis_radio_options,
                    value='who_won',
                    labelStyle={'display': 'inline-block',
                                'margin-right': '10px'}
                ),
                dcc.Graph(id='elections_barplot')], style={
                    'display': 'inline-block', 'width': '30%', 'verticalAlign': 'top',
                'minWidth': '200px', 'margin-right': '2%'}),
            html.Div([
                dl.Map([
                    dl.TileLayer(
                        url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                    dl.LocateControl(
                        locateOptions={'enableHighAccuracy': True}),
                    dl.GeoJSON(id='stats_layer', data=stats_data,
                               hoverStyle=hover_style,
                               style=won_style_handle,
                               zoomToBoundsOnClick=True,
                               hideout=dict(
                                   colorscale=colorscale, classes=classes, style=style, hoverStyle=hover_style, colorProp="max_label")
                               ),
                    dl.Colorbar(id='colorbar', position='bottomright', opacity =0, tickText=['','']),
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

            ], style={'display': 'inline-block', 'width': '60%', 'verticalAlign': 'top', 'minWidth': '600px', 'margin-left': '2%'}
            )
        ],
    ),

])


@ app.callback(Output("info", "children"), Input("stats_layer", "hoverData"))
def info_hover(feature):
    return get_info(feature)


@ app.callback(Output('elections_barplot', 'figure'), Input('stats_layer', 'clickData'), State('elections_barplot', 'figure'))
def update_barplot(clickData, fig):

    # if raio_map_analysis == 'who_won':
    if fig is None:
        fig = generate_barplot()
        return fig
    if clickData is None:
        return fig
    else:
        return generate_barplot(clickData)


@ app.callback(Output('env_map', 'children'), Input('env_map', 'children'), State('stats_layer', 'data'), Input('stats_layer', 'clickData'), Input('raio_map_analysis', 'value'))
def update_map(map_layers, map_json, clickData, radio_map_option):
    hideout = dict(colorscale=colorscale, classes=classes,
                style=style, hoverStyle=hover_style, colorProp="max_label")
    no_data = False
    if clickData is not None:
        feature_id = clickData["properties"]["YISHUV_STAT11"]
        stats_map_data_gdf= get_kdtree(stat_filter=feature_id, gdf=stats_data_gdf.copy())
        stats_data = stats_map_data_gdf.__geo_interface__
    else:
        stats_data = map_json
        no_data = True
    
    if no_data == False:
        if radio_map_option =='who_won':
            print('who won')
            map_layers = [dl.TileLayer(url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                            dl.LocateControl(locateOptions={'enableHighAccuracy': True}),
                            dl.GeoJSON(id='stats_layer', data=stats_data,
                                    hoverStyle=hover_style,
                                    style=won_style_handle,
                                    zoomToBoundsOnClick=True,
                                    hideout=hideout,
                            ),
                            info
                            ]
        elif radio_map_option == 'kdtree':
            print('kde Tree')
            hideout['colorscale'] = kde_colorscale
            hideout['classes'] = kde_classes
            hideout['colorProp'] = 'kde_distnace'
            min_, max_ = stats_map_data_gdf['kde_distnace'].min(), stats_map_data_gdf['kde_distnace'].max()
            classes_colormap = np.linspace(min_, max_, num=8)
            ctg = [f"{round(cls,1)}+" for i, cls in enumerate(classes_colormap[:-1])] + [f"{round(classes_colormap[-1],1)}+"]
            colorbar = dlx.categorical_colorbar(categories= ctg,colorscale=kde_colorscale, width=500, height=30, position="bottomright")
            map_layers = [dl.TileLayer(url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                        dl.LocateControl(locateOptions={'enableHighAccuracy': True}),
                        dl.GeoJSON(id='stats_layer', data=stats_data,
                                hoverStyle=hover_style,
                                style=kde_style_handle,
                                zoomToBoundsOnClick=True,
                                hideout=hideout,
                        ),
                        colorbar,
                        info
                        ]
            return map_layers

    return map_layers

if __name__ == '__main__':
    app.run_server(debug=True)
