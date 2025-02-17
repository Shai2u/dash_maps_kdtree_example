"""
Author: Shai Sussman
Date: 2025-02-17
Description: This application provides an interactive visualization of Israeli election data using Dash.
It includes features for analyzing voting patterns through KDTree clustering, KMeans clustering,
and geographic distance relationships. The app displays results on an interactive map with 
accompanying statistical plots and allows users to explore spatial relationships between different
voting districts.

Key Features:
- Interactive map visualization of election results
- KDTree and KMeans clustering analysis
- Distance-based relationship plots
- Geographic data exploration tools
"""

from dash import Dash, dcc, html, Input, Output, State
import dash_leaflet as dl
import dash_leaflet.express as dlx
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import KDTree, distance
from sklearn.cluster import KMeans
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import os
from app_helper import won_style_handle, style, hover_style, map_analysis_radio_options, kde_classes, kde_colorscale, kde_style_handle, kmeans_color_dict, kmeans_style_handle

pd.options.display.max_columns = 150

def load_data(path: str, **kwargs) -> pd.DataFrame:
    """Load data from CSV or GeoJSON file.

    Parameters
    ----------
    path : str
        Path to the data file. Must be either .csv or .geojson
    **kwargs : dict
        Additional keyword arguments. For GeoJSON files, can include 'crs' 
        to specify coordinate reference system.

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        DataFrame containing the loaded data. Returns GeoDataFrame if input is GeoJSON.

    Raises
    ------
    ValueError
        If file type is not supported (.csv or .geojson)
    """
    if path.endswith('.csv'):
        if 'index_col' in kwargs:
            return pd.read_csv(path, index_col=kwargs['index_col'])
        else:   
            return pd.read_csv(path)
    elif path.endswith('.geojson'):
        if 'crs' in kwargs:
            return gpd.read_file(path).to_crs(kwargs['crs'])
        else:   
            return gpd.read_file(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def load_data_main() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load Hebrew-English dictionary and statistical population data.

    Loads two data files:
    1. Hebrew-English dictionary mapping Hebrew column names to English
    2. Statistical population data with voting information from 2022

    Returns
    -------
    tuple[pd.DataFrame, gpd.GeoDataFrame]
        heb_dict_df : pd.DataFrame
            DataFrame containing Hebrew to English column name mappings
        stats_data_original_gdf : gpd.GeoDataFrame 
            GeoDataFrame containing statistical population and voting data
            with geometry in EPSG:4326 projection
    """
    heb_dict_df = load_data('data/heb_english_dict.csv', index_col=0)
    stats_data_original_gdf = load_data('data/stat_pop_simpl_votes_2022.geojson', crs='EPSG:4326')
    return heb_dict_df, stats_data_original_gdf

def setup_col_rename_color_dicts(heb_dict_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Set up dictionaries for column renaming and party colors.

    Creates three dictionaries from the Hebrew-English dictionary DataFrame:
    1. Column name mapping from Hebrew to English
    2. Color mapping for parties by index
    3. Color mapping for parties by name

    Parameters
    ----------
    heb_dict_df : pd.DataFrame
        DataFrame containing Hebrew-English mappings and color codes
        Must have columns for Hebrew names, English names, and color codes

    Returns
    -------
    tuple[dict, dict, dict]
        col_rename : dict
            Maps Hebrew column names to English names
        color_dict_party_index : dict 
            Maps party indices to color codes
        color_dict_party_name : dict
            Maps party names to color codes
    """
    col_rename = heb_dict_df.T.set_index(0)[1].to_dict()
    color_dict_party_index = heb_dict_df.T.set_index(0).loc['party_1':][2].to_dict()
    color_dict_party_name = heb_dict_df.T.set_index(1).loc['labor':][2].to_dict()   
    return col_rename, color_dict_party_index, color_dict_party_name


def get_kdtree(gdf: gpd.GeoDataFrame, stat_filter: str) -> gpd.GeoDataFrame:

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


# RUN KMENAS ONLY When The number of cluster is changed!!!!!!
def get_kmeans_cluster_add_column(n_cluster, stats_map_data_gdf):
    gdf = stats_map_data_gdf.copy()
    df = gdf.copy()
    df = df.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
            'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'], axis=1).copy()
    
    # 3. Normalize the data
    df = df.apply(
        lambda p: p/p['bzb'], axis=1).drop('bzb', axis=1)
    df_kmeans = df.copy()

    for col in ['votes', 'invalid_votes', 'valid_votes','kde_distance', 'cluster', 'id']:
        if col in df_kmeans.columns:
            df_kmeans.drop(col, inplace=True, axis=1)
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    # Fit the model
    kmeans.fit(df_kmeans.values)

    # Get the cluster labels
    df['cluster'] = kmeans.labels_
    gdf['cluster'] = kmeans.labels_
    return df, gdf,  kmeans



def get_info(feature, col_rename):
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


def build_near_clsuter_bar_fig(gdf_sorted, kdtree_distance):
    fig = px.bar(gdf_sorted, x='name_stat', y='kde_distance', title=f'Top {kdtree_distance} most similar voting pattern', custom_data=['YISHUV_STAT11'], barmode='group')
    fig.update_layout(
        xaxis_tickangle=-90,
        yaxis=dict(visible=True),
        height=550,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=0, r=0, t=15, b=0), 
        font=dict(size=14),
        title_y=0.98, 
        )
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    return fig


def generate_barplot(feature=None):
    print('feature: ', feature)
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]
    else:
        return {}
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
        'x': '', 'y': ''}, color=categories, color_discrete_map=color_dict_party_name, title=stat_name)
    fig.update_layout(
        xaxis_tickangle=-90,
        yaxis=dict(range=[0, top_ten.max()+0.1 if top_ten.max()
                   > 0.5 else 0.5], visible=False),
        height=300,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), title_y=0.9, title_x=0.6, font=dict(size=14))
    return fig


def generate_histogram_with_line(df_kmeans, eu_distance):

    # Plot histogram of df_kmeans distances using Plotly
    # fig = px.histogram(df_kmeans, x='distance_to_cluster', nbins=30, title='Histogram of Distances to Cluster Center')

    # x = fig.data[0].x
    # Create 30 bins
    hist, bin_edges = np.histogram(df_kmeans['distance_to_cluster'], bins=30)

    # Get the most frequent bin
    max_y = np.max(hist)
    # Create a bar plot using hist and bin_edges
    fig = px.bar(x=bin_edges[:-1], y=hist, labels={'x': 'Distance to Cluster Center', 'y': 'Frequency'}, title='Histogram of Distances to Cluster Center')
    fig.update_traces(marker_color='blue', marker = {'line':{'width':0}})

    # Add a red vertical line at 0.3
    fig.add_shape(
        type='line',
        x0=eu_distance, y0=0, x1=eu_distance, y1=max_y,
        line=dict(color='red', dash='dash', width=2)
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Distance to Cluster Center',
        yaxis_title='Frequency',
        template='plotly_white',
        margin=dict(l=0, r=0, t=0, b=0), 
        title_y=0.9, 
        title_x=0.6, 
        font=dict(size=14),
        bargap=0.1,
        height=300,

    )
    

    return fig

def get_kmeans_euclidian_distance(df, filter_row, kmeans):
    filter_row = (filter_row/filter_row['bzb']).drop('bzb')
    for col in ['votes', 'invalid_votes', 'valid_votes','kde_distance', 'cluster', 'id']:
        if col in filter_row.index:
            filter_row.drop(col, inplace=True)
    selected_cluster = kmeans.predict(filter_row.values.reshape(1, -1))[0]
    df_subset = df[df['cluster'] == selected_cluster].drop(columns='cluster').copy()
    if 'cluster' in df_subset.columns:
        df_subset.drop(columns='cluster', inplace=True)
    # Get the attributes of the selected center cluster
    print('Selected Cluster', selected_cluster)
    selected_cluster_attributes = kmeans.__dict__['cluster_centers_'][selected_cluster]
        
    df_subset = df_subset.apply(lambda p: p/p['bzb'], axis=1)

    # Build the KDTree for the single cluster subset
    for col in ['bzb', 'votes', 'invalid_votes', 'valid_votes', 'kde_distance', 'cluster', 'id']:
        if col in df_subset.columns:
            df_subset.drop(col, inplace=True, axis=1)

    tree = KDTree(df_subset.values)

    # Query KDTree for the 3 nearest neighbors (including the row itself)
    distances, indices = tree.query(selected_cluster_attributes, k=len(df_subset))

    # Retrieve nearest rows using the indices
    df_kmeans = df_subset.iloc[indices].copy()
    df_kmeans['distance_to_cluster'] = distances
    eu_distance = distance.euclidean(selected_cluster_attributes, filter_row.values)
    return df_kmeans, eu_distance

def process_stats_data(stats_data_original_gdf: gpd.GeoDataFrame, col_rename: dict) -> gpd.GeoDataFrame:
    """
    Process the statistical data GeoDataFrame by renaming columns and cleaning station names.

    Parameters
    ----------
    stats_data_original_gdf : geopandas.GeoDataFrame
        The original GeoDataFrame containing statistical data with Hebrew column names
    col_rename : dict
        Dictionary mapping Hebrew column names to English column names

    Returns
    -------
    geopandas.GeoDataFrame
        The processed GeoDataFrame with renamed columns and cleaned station names

    Notes
    -----
    This function performs two main operations:
    1. Renames columns using the provided mapping dictionary
    2. Removes 'No Name' text from station names in the 'sta_22_names' column
    """
    stats_data_original_gdf.rename(columns=col_rename, inplace=True)
    stats_data_original_gdf['sta_22_names'] = stats_data_original_gdf['sta_22_names'].str.replace('No Name', '')
    return stats_data_original_gdf

### Load the data
heb_dict_df, stats_data_original_gdf = load_data_main()

#### Setting up dictionaries and classes for the map
col_rename, color_dict_party_index, color_dict_party_name = setup_col_rename_color_dicts(heb_dict_df)

# Prepare spatial data and convert Hebrew column names to English using the dictionary
stats_data_original_gdf = process_stats_data(stats_data_original_gdf, col_rename)


stats_data_gdf = gpd.GeoDataFrame()
stats_data_gdf = get_kdtree(stat_filter=stats_data_original_gdf.sample(1)['YISHUV_STAT11'].values[0], gdf=stats_data_original_gdf.copy())
stats_data = stats_data_gdf.__geo_interface__

info = html.Div(children=get_info(feature=None, col_rename=col_rename), id="info", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"})



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
                html.Div([dcc.Graph(id='kmeans_distance_barplot'), dcc.Graph(id='kmeans_scatterplot')],id='kmeans_frequencybarplot_div', style={'display':'none'}) ], style={
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
                                   color_dict=color_dict_party_index, style=style, hoverStyle=hover_style, win_party="max_label")
                               ),
                    dl.Colorbar(id='colorbar', position='bottomright', opacity =0, tickText=['','']),
                    info
                ],
                    center=[32, 34.9],
                    zoom=12,
                    style={'height': '99vh'},
                    id='env_map',
                    dragging=True,
                    zoomControl=True,
                    scrollWheelZoom=True,
                    doubleClickZoom=True,
                    boxZoom=True,
                )

            ], style={'display': 'inline-block', 'width': '60%', 'verticalAlign': 'top', 'margin-left': '2%'}
            )
        ],
    ),
    dcc.Store(id='temp-data-store'),

])


@ app.callback(Output("info", "children"), Input("stats_layer", "hoverData"))
def info_hover(feature):
    return get_info(feature = feature, col_rename=col_rename)

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


@ app.callback(Output('env_map', 'children'), Output('temp-data-store', 'data'), Input('env_map', 'children'), State('stats_layer', 'data'), Input('stats_layer', 'clickData'), Input('raio_map_analysis', 'value'), Input('near_cluster', 'value'), 
Input('kmeans_cluster', 'value'))
def update_map(map_layers, map_json, clickData, radio_map_option, kdtree_distance, kmeans_cluster):
    hideout = {"color_dict":color_dict_party_index, "style":style, "hoverStyle":hover_style, 'win_party':"max_label"}
    no_data = False
    data_store_temp = {}
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
            return map_layers, data_store_temp
        
        else:
            hideout['color_dict'] = kmeans_color_dict
            hideout['clusters_col'] = 'cluster'
            # Check if kmeans_cluster value has changed from previous run
            flag_run_kmeans = False
            if data_store_temp.get('model_stored') and os.path.exists(data_store_temp.get('model_stored')):
                previous_model = joblib.load(data_store_temp.get('model_stored'))
                if previous_model.n_clusters != kmeans_cluster:
                    # If the number of clusters has changed, flag run the kmeans again
                    flag_run_kmeans = True
                else:
                    gdf = stats_data_gdf.copy()
                    kmeans = previous_model
            else:
                flag_run_kmeans = True
            # Run kmeans if the number of clusters has changed
            if flag_run_kmeans:
                _, gdf,  kmeans =  get_kmeans_cluster_add_column(kmeans_cluster, stats_data_gdf.copy())


            # Get the attributes of the KMeans instance

            stats_data = gdf.__geo_interface__
            
            map_layers = [dl.TileLayer(url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                        dl.LocateControl(locateOptions={'enableHighAccuracy': True}),
                        dl.GeoJSON(id='stats_layer', data=stats_data,
                                hoverStyle=hover_style,
                                style=kmeans_style_handle,
                                zoomToBoundsOnClick=True,
                                hideout=hideout,
                        ),
                        info
                        ]

            
            # Store the kmeans model in dcc.Store for later use
            # Serialize the model to a byte stream
            joblib.dump(kmeans, 'kmeans_model.joblib')


            # Store the byte stream in a variable
            data_store_temp = {'model_stored':'kmeans_model.joblib'}

    return map_layers, data_store_temp



@ app.callback(Output('kde_distance_barplot', 'figure'), Input('stats_layer', 'data'), Input('near_cluster', 'value'))
def update_near_clster_bar(map_json, kdtree_distance):
    # Convert GeoJSON data to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(map_json['features'])
    gdf = gdf[gdf['kde_distance']>0].reset_index(drop=True)
    # Generate a barplot based on the KDE distances
    gdf_sorted = gdf.sort_values(by='kde_distance').iloc[0:kdtree_distance]
    # gdf_sorted['name_stat'] = gdf_sorted['Shem_Yishuv'] + '-' + gdf_sorted['sta_22_names']
    gdf_sorted['name_stat'] = gdf_sorted.apply(lambda p: p['Shem_Yishuv']+'-'+ p['sta_22_names'] if len(p['sta_22_names'])>0 else  p['Shem_Yishuv']+'-' + str(p['YISHUV_STAT11'])[-3:], axis=1) 

    fig_kde = build_near_clsuter_bar_fig(gdf_sorted, kdtree_distance)
    
    return fig_kde
    # Generate a sample barplot

@ app.callback(Output('kmeans_distance_barplot', 'figure'), Output('kmeans_scatterplot', 'figure'), State('stats_layer', 'data'), Input('stats_layer', 'clickData'), State('temp-data-store', 'data'), State('kmeans_distance_barplot', 'figure'), State('kmeans_scatterplot', 'figure'), prevent_initial_call=True)
def update_kmeans_distance_bar(map_json, feature, saved_model, fig_bar, fig_scatter):
    # Prevent callback execution on initial load
    if map_json is None or feature is None:
        if None in [fig_bar, fig_scatter]:
            return {}, {}
        else:
            return fig_bar, fig_scatter
    
    feature_id = np.random.choice(stats_data_gdf['YISHUV_STAT11'].values)
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]
    gdf = gpd.GeoDataFrame.from_features(map_json['features'])
    if 'cluster' not in gdf.columns:
        return {}, {}
    if 0 not in gdf['cluster'].unique():
        return {}, {}
    
    if saved_model == {} or saved_model is None:
        return {}, {}
    else:
        kmeans_model = joblib.load(saved_model['model_stored'])

        df = gdf.copy()
        df = df.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
            'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'], axis=1).copy()
    
    # df, gdf,  kmeans_model =  get_kmeans_cluster_add_column(kmeans_cluster, gdf)
    df_copy = gdf.copy()
    feature_index_id = df_copy[df_copy['YISHUV_STAT11'] == feature_id].index[0]
    kdf_filter_row = df_copy.loc[feature_index_id]
    kdf_filter_row = kdf_filter_row.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
                'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label']).copy()
    
    df_kmeans, eu_distance = get_kmeans_euclidian_distance(df , kdf_filter_row, kmeans_model)
    fig_bar = generate_histogram_with_line(df_kmeans, eu_distance)
    
    gdf_centroids = gdf.copy().assign(geometry=gdf.centroid)
    centro_filter_row = gdf_centroids.loc[df_kmeans['distance_to_cluster'].idxmin(), ['geometry', 'cluster']]
    gdf_filter_cluster = gdf_centroids.iloc[df_kmeans.index].copy()[['geometry', 'cluster']]
    # bookmark kde tree
    centro_filter_row['x'] = centro_filter_row['geometry'].x
    centro_filter_row['y'] = centro_filter_row['geometry'].y
    centro_filter_row = centro_filter_row.drop(['cluster', 'geometry'])

    gdf_filter_cluster['x'] = gdf_filter_cluster['geometry'].x
    gdf_filter_cluster['y'] = gdf_filter_cluster['geometry'].y
    gdf_filter_cluster.drop(columns = ['cluster', 'geometry'], inplace=True)

    # this is wrong! I should take the closest value to cluster and that should messure the distnace to all the other points, and than take the selectd zone and check it's location
    # FIX THIS!
    tree = KDTree(gdf_filter_cluster.values)

    # Query KDTree for the 3 nearest neighbors (including the row itself)
    distances, indices = tree.query(centro_filter_row.values, k=len(gdf_filter_cluster))
    gdf_filter_cluster = gdf_filter_cluster.iloc[indices].copy()
    gdf_filter_cluster['geo_distance'] = distances
    
    gdf_filter_cluster = gdf_filter_cluster.sort_index()
    df_kmeans = df_kmeans.sort_index()

    # concat the geo distance (mulitipied by 111 to convert to km) and the distance to cluster
    kmeans_geo_distance = pd.concat([gdf_filter_cluster[['geo_distance']]*111, df_kmeans[['distance_to_cluster']]], axis=1)
    selected_feature_distances_dict = kmeans_geo_distance.loc[feature_index_id].to_dict()
    # feature_index_id
    # New scatterplot figure comes here
    fig_scatter = px.scatter(
        kmeans_geo_distance, 
        x='distance_to_cluster', 
        y='geo_distance',
        labels={
            'distance_to_cluster': 'Distance to Cluster Center (no units)',
            'geo_distance': 'Geo Distance (in KM)'
        },
        title='Distance Relationships'
    )

    # Add crosshair lines for the selected feature
    fig_scatter.add_shape(
        type='line',
        x0=selected_feature_distances_dict['distance_to_cluster'],
        y0=0,
        x1=selected_feature_distances_dict['distance_to_cluster'], 
        y1=kmeans_geo_distance['geo_distance'].max(),
        line=dict(color='red', width=1, dash='solid'),
        name='Selected Area'
    )
    fig_scatter.add_shape(
        type='line',
        x0=0,
        y0=selected_feature_distances_dict['geo_distance'],
        x1=kmeans_geo_distance['distance_to_cluster'].max(),
        y1=selected_feature_distances_dict['geo_distance'],
        line=dict(color='red', width=1, dash='solid'),
        name='Selected Area'
    )

    fig_scatter.update_layout(
        template='plotly_white',
        height=300,
        showlegend=True
    )
    # Calculate polynomial regression
    x = kmeans_geo_distance['distance_to_cluster']
    y = kmeans_geo_distance['geo_distance']
    
    # Fit polynomial of degree 2
    coeffs = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coeffs)
    
    # Calculate R-squared
    y_pred = polynomial(x)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
    
    # Generate points for smooth curve
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = polynomial(x_smooth)
    
    # Add regression line and R² annotation
    fig_scatter.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            showlegend=False,
            line=dict(color='green', width=2)
        )
    )
    
    # Add R² annotation above the regression line
    fig_scatter.add_annotation(
        x=x_smooth.mean(),
        y=y_smooth.mean(),
        text=f'R² = {r2:.3f}',
        showarrow=False,
        yshift=20,
        font=dict(size=12)
    )
    fig_scatter.update_xaxes(range=[0, x.max()])
    fig_scatter.update_yaxes(range=[0, y.max()])
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=0, b=0), title_y=0.9, title_x=0.6, font=dict(size=14))

    #### !!!! RETURTN DCCC.GRAP
    return fig_bar, fig_scatter
    
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
