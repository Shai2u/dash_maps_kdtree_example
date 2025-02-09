from dash import Dash, dcc, html, Input, Output, State
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
from scipy.spatial import KDTree, distance
import geopandas as gpd
import pandas as pd
from app_helper import won_style_handle, style, hover_style, map_analysis_radio_options, kde_classes, kde_colorscale, kde_style_handle
import numpy as np
import plotly.express as px
import json
from sklearn.cluster import KMeans


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
    gdf_kde['kde_distance'] = distances
    return gdf_kde


def get_kmeans_cluster_add_column(n_cluster, stats_map_data_gdf):
    df = stats_map_data_gdf.copy()
    df = df.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
            'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'], axis=1).copy()
    
    # 3. Normalize the data
    df = df.apply(
        lambda p: p/p['bzb'], axis=1).drop('bzb', axis=1)
    df_kmeans = df.copy()
    for col in ['kde_distance', 'cluster', 'id']:
        if col in df_kmeans.columns:
            df_kmeans.drop(col, inplace=True, axis=1)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    # Fit the model
    kmeans.fit(df_kmeans.values)

    # Get the cluster labels
    df['cluster'] = kmeans.labels_
    return df, kmeans

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


def build_near_clsuter_bar_fig(gdf_sorted, kdtree_distance):
    fig = px.bar(gdf_sorted, x='name_stat', y='kde_distance', title=f'Top {kdtree_distance} most similar voting pattern', custom_data=['YISHUV_STAT11'], barmode='group')
    fig.update_layout(
        xaxis_tickangle=-90,
        yaxis=dict(visible=True),
        height=500,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), font=dict(size=14))
    return fig


def generate_barplot(feature=None):
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]
    else:
        feature_id = np.random.choice(stats_data_gdf['YISHUV_STAT11'].values)
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
        height=500,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), title_y=0.9, title_x=0.95, font=dict(size=14))
    return fig


def generate_histogram_with_line(df_kmeans, eu_distance):

    # Plot histogram of df_kmeans distances using Plotly
    # fig = px.histogram(df_kmeans, x='distance_to_clsuter', nbins=30, title='Histogram of Distances to Cluster Center')

    # x = fig.data[0].x
    # Create 30 bins
    hist, bin_edges = np.histogram(df_kmeans['distance_to_clsuter'], bins=30)

    # Get the most frequent bin
    max_y = np.max(hist)
    # Create a bar plot using hist and bin_edges
    fig = px.bar(x=bin_edges[:-1], y=hist, labels={'x': 'Distance to Cluster Center', 'y': 'Frequency'}, title='Histogram of Distances to Cluster Center')
    fig.update_traces(marker_color='blue', marker = {'line':{'width':0}})
    fig.update_layout(bargap=0.1)

    # Add a red vertical line at 0.3
    fig.add_shape(
        type='line',
        x0=eu_distance, y0=0, x1=eu_distance, y1=max_y,
        line=dict(color='red', dash='dash', width=2)
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Distance to Cluster Center',
        yaxis_title='Frequency'
    )

    return fig

def get_kmeans_histogram_with_selected_line(df, filter_row, kmeans):
    filter_row = (filter_row/filter_row['bzb']).drop('bzb')
    for col in ['kde_distance', 'cluster', 'id']:
        if col in filter_row.index:
            filter_row.drop(col, inplace=True)
    selected_cluster = kmeans.predict(filter_row.values.reshape(1, -1))[0]
    df_subset = df[df['cluster'] == selected_cluster].drop(columns='cluster').copy()
    if 'cluster' in df_subset.columns:
        df_subset.drop(columns='cluster', inplace=True)
    # Get the attributes of the selected center cluster
    selected_cluster_attributes = kmeans.__dict__['cluster_centers_'][selected_cluster]
        

    # Build the KDTree for the single cluster subset
    for col in ['kde_distance', 'cluster', 'id']:
        if col in df_subset.columns:
            df_subset.drop(col, inplace=True, axis=1)

    tree = KDTree(df_subset.values)

    # Query KDTree for the 3 nearest neighbors (including the row itself)
    distances, indices = tree.query(selected_cluster_attributes, k=len(df_subset))

    # Retrieve nearest rows using the indices
    df_kmeans = df_subset.iloc[indices].copy()
    df_kmeans['distance_to_clsuter'] = distances
    eu_distance = distance.euclidean(selected_cluster_attributes, filter_row.values)
    fig = generate_histogram_with_line(df_kmeans, eu_distance)
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
                html.Div([
                html.Div(dcc.RadioItems(
                    id='raio_map_analysis',
                    options=map_analysis_radio_options,
                    value='who_won',
                    labelStyle={'display': 'inline-block',
                                'margin-right': '10px'}
                )),
                html.Div(dcc.Slider(
                    id='near_cluster',
                    min=5,
                    max=100,
                    step=1,
                    value=25,
                    marks={i: str(i) for i in range(10, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},

                ),id='near_cluster_div', style={'width': '60%'}),
                 html.Div(dcc.Slider(
                    id='kmeans_cluster',
                    min=2,
                    max=10,
                    step=1,
                    value=4,
                    marks={i: str(i) for i in range(2, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),id='kmeans_cluster_div', style={'width': '60%', 'display':'none'}),

                ],style={
                    'display': 'flex', 'width': '100%', 'justify-content': 'space-between'}),
                dcc.Graph(id='elections_barplot'), 
                html.Div(dcc.Graph(id='kde_distance_barplot'),id='kde_distance_barplot_div', style={'display':'none'}),
                html.Div(dcc.Graph(id='kmeans_distance_barplot'),id='kmeans_frequencybarplot_div', style={'display':'none'}) ], style={
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
                                   color_dict=colors_dict, style=style, hoverStyle=hover_style, win_party="max_label")
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

@ app.callback(Output("near_cluster_div", "style"), Output("kmeans_cluster_div", "style"), Output("kde_distance_barplot_div", "style"), Output("kmeans_frequencybarplot_div","style"), Input('raio_map_analysis', 'value'))
def controller(radioButton):
    if radioButton == 'who_won':
        return [{'display':'none'},{'display':'none'}, {'display':'none'}, {'display':'none'}]
    elif radioButton == 'kdtree':
        return [{'width': '60%', 'display':'block'}, {'display':'none'}, {'display':'block'}, {'display':'none'}]
    else:
        return [{'display':'none'}, {'width': '60%', 'display':'block'},{'display':'none'}, {'display':'block'}]
        #Add This display kmeans_frequencybarplot_div

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


@ app.callback(Output('env_map', 'children'), Input('env_map', 'children'), State('stats_layer', 'data'), Input('stats_layer', 'clickData'), Input('raio_map_analysis', 'value'), Input('near_cluster', 'value'))
def update_map(map_layers, map_json, clickData, radio_map_option, kdtree_distance):
    hideout = {"color_dict":colors_dict, "style":style, "hoverStyle":hover_style, 'win_party':"max_label"}
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
            hideout['colorscale'] = kde_colorscale
            hideout['classes'] = kde_classes
            hideout['colorProp'] = 'kde_distance'
            gdf = get_kdtree(stat_filter=feature_id, gdf=stats_data_gdf.copy())
            gdf = gdf.sort_values(by='kde_distance').reset_index(drop=True)
            gdf = gdf.iloc[0:kdtree_distance+1]
            min_, max_ = gdf['kde_distance'].min(), gdf['kde_distance'].max()
            stats_data = gdf.__geo_interface__
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
        
        else:
            print('kmeans')
            # gdf, kmeans = get_kmeans(kmeans_cluster, stats_map_data_gdf)
            # Get the attributes of the KMeans instance



    return map_layers



@ app.callback(Output('kde_distance_barplot', 'figure'), Input('stats_layer', 'data'), Input('near_cluster', 'value'))
def update_near_clster_bar(map_json, kdtree_distance):
    # Convert GeoJSON data to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(map_json['features'])
    gdf = gdf[gdf['kde_distance']>0].reset_index(drop=True)
    # Generate a barplot based on the KDE distances
    gdf_sorted = gdf.sort_values(by='kde_distance').iloc[0:kdtree_distance]
    # gdf_sorted['name_stat'] = gdf_sorted['Shem_Yishuv'] + '-' + gdf_sorted['sta_22_names']
    gdf_sorted['name_stat'] = gdf_sorted.apply(lambda p: p['Shem_Yishuv']+'-'+ p['sta_22_names'] if len(p['sta_22_names'])>0 else  p['Shem_Yishuv']+'-' + str(p['YISHUV_STAT11'])[-3:], axis=1) 

    fig = build_near_clsuter_bar_fig(gdf_sorted, kdtree_distance)
    
    return fig
    # Generate a sample barplot

@ app.callback(Output('kmeans_distance_barplot', 'figure'), Input('stats_layer', 'data'), Input('stats_layer', 'clickData'), Input('kmeans_cluster', 'value'))
def update_kmeans_distance_bar(map_json, feature, kmeans_cluster):
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]
    else:
        feature_id = np.random.choice(stats_data_gdf['YISHUV_STAT11'].values)
    print('kmeans before figure')
    gdf = gpd.GeoDataFrame.from_features(map_json['features'])
    df, kmeans =  get_kmeans_cluster_add_column(kmeans_cluster, gdf)
    df_copy = stats_data_gdf.copy()
    kdf_filter_row = df_copy[df_copy['YISHUV_STAT11'] == feature_id].iloc[0]
    kdf_filter_row = kdf_filter_row.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
                'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label']).copy()
    

    fig = get_kmeans_histogram_with_selected_line(df , kdf_filter_row, kmeans)
    print(str(fig))
    print('kmeans fig')
    return fig
    
@ app.callback(Output('env_map', 'viewport'), Input('kde_distance_barplot', 'clickData'), prevent_initial_call=True)

def zoom_to_feature_by_bar(clickData):
    if clickData is not None:
        stat = clickData['points'][0]['customdata'][0]
        centroid = stats_data_original_gdf[stats_data_original_gdf['YISHUV_STAT11'] == stat].iloc[0]['geometry'].centroid
        return dict(center=[centroid.y, centroid.x], zoom=15, transition="flyTo")
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)
