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

def get_kdtree_distance(gdf: gpd.GeoDataFrame, feature: str) -> gpd.GeoDataFrame:
    """Build KD-tree from voting data and find nearest neighbors to selected feature.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing voting data and geometries
    feature : str
        Dictionary containing properties of selected feature including YISHUV_STAT11 ID

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing all features sorted by distance to selected feature,
        with added 'kde_distance' column

    Notes
    -----
    The function:
    1. Extracts the row for the selected feature
    2. Drops non-numeric columns from both selected row and full dataset
    3. Normalizes voting data by dividing by total votes ('bzb')
    4. Builds KD-tree from normalized data
    5. Finds distances to all other features
    6. Returns original GeoDataFrame with added distances
    """
    # Get the row for the selected feature
    kdf_filter_row = gdf[gdf['YISHUV_STAT11'] == feature["properties"]["YISHUV_STAT11"]].iloc[0]
    
    # Drop non-numeric columns from both selected row and full dataset
    labels_to_drop= ['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
                       'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label']
    kde_df = gdf.drop(labels_to_drop, axis=1).copy()
    kdf_filter_row = kdf_filter_row.drop(labels_to_drop)

    # Normalize the data
    kde_df = kde_df.apply(
        lambda p: p/p['bzb'], axis=1).drop('bzb', axis=1)
    kdf_filter_row = (kdf_filter_row/kdf_filter_row['bzb']).drop('bzb')

    # Build the KDTree
    tree = KDTree(kde_df.values)

    # Query KDTree for the 3 nearest neighbors (including the row itself)
    distances, indices = tree.query(kdf_filter_row.values, k=len(kde_df))

    # Retrieve nearest rows using the indices
    gdf_kde = gdf.iloc[indices].copy()
    gdf_kde['kde_distance'] = distances
    return gdf_kde

def get_kmeans_cluster_add_column(n_cluster, stats_map_data_gdf):
    gdf = stats_map_data_gdf.copy()
    df = gdf.copy()
    df = df.drop(['geometry', 'YISHUV_STAT11', 'Shem_Yishuv_English',
            'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'], axis=1).copy()
    
    # Normalize the data
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
    """Build bar plot showing nearest neighbors based on KDE distances.

    Parameters
    ----------
    gdf_sorted : pd.DataFrame
        DataFrame containing sorted features by KDE distance
    kdtree_distance : int
        Number of nearest neighbors to show

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Bar plot figure showing nearest neighbors by KDE distance
    """
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
    fig.update_xaxes(title_text='Statistical Area')
    fig.update_yaxes(title_text='KDE Distance')

    return fig

def generate_election_barplot_fig(feature=None):
    """Generate bar plot showing top 8 parties by votes.

    Parameters
    ----------
    feature : dict, optional
        A dictionary representing a feature with properties. Default is None.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Bar plot figure showing top 6 parties by votes

    """
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]
    else:
        return {}
    # Select statistical area and select only the relevant columns, sort by votes
    selected_row = stats_data_original_gdf[stats_data_original_gdf['YISHUV_STAT11']
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
    top_ten = pd.Series(percent[0:8], index=selected_row.index[0:8])
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
        height=350,
        showlegend=False,
        template='plotly_white',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=0, r=0, t=0, b=0),
        title_y=0.9,
        title_x=0.6,
        font=dict(size=14)
    )
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    return fig

def generate_histogram_with_line(df_kmeans, eu_distance):
    """Generate histogram with vertical line at Euclidean distance.

    Parameters
    ----------
    df_kmeans : pd.DataFrame
        DataFrame containing cluster distances
    eu_distance : float
        Euclidean distance

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Histogram figure with vertical line at Euclidean distance

    """
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
        height=350,
    )
    
    return fig

def get_kmeans_euclidian_distance(df, filter_row, kmeans):
    """Get Euclidean distance between selected cluster and all other clusters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cluster data
    filter_row : pd.Series
        Series containing filter row data
    kmeans : sklearn.cluster.KMeans
        KMeans model

    Returns
    -------
    tuple
        Tuple containing:
    """
    # Normalize the filter row by the total votes
    filter_row = (filter_row/filter_row['bzb']).drop('bzb')
    # Drop the columns that are not needed and check if they exist
    for col in ['votes', 'invalid_votes', 'valid_votes','kde_distance', 'cluster', 'id']:
        if col in filter_row.index:
            filter_row.drop(col, inplace=True)
    # Get the attributes of the selected center cluster
    selected_cluster = kmeans.predict(filter_row.values.reshape(1, -1))[0]
    # Drop the cluster column
    df_subset = df[df['cluster'] == selected_cluster].drop(columns='cluster').copy()
    if 'cluster' in df_subset.columns:
        df_subset.drop(columns='cluster', inplace=True)
    # Get the attributes of the selected center cluster
    selected_cluster_attributes = kmeans.__dict__['cluster_centers_'][selected_cluster]
    # Normalize the dataframe by the total votes
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


def _prepare_map_layers_for_winner(stats_data, hideout):
    """Prepare map layers for winner analysis.

    Parameters
    ----------
    stats_data : dict
        Dictionary containing statistical data
    hideout : dict
        Dictionary containing map settings

    Returns
    -------
    list
        List of map layers

    Notes
    -----
    The function creates a list of map layers including:
    - Tile layer for map tiles
    - Locate control for user location
    - GeoJSON layer for statistical data
    - Info component for feature details    
    """
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
    return map_layers

def _prepare_map_layers_for_kdtree(stats_data, hideout, colorbar):
    """Prepare map layers for k-d tree analysis.

    Parameters
    ----------
    stats_data : dict
        Dictionary containing statistical data
        hideout : dict
        Dictionary containing map settings
    colorbar : dict
        Dictionary containing colorbar settings

    Returns
    -------
    list
        List of map layers

    Notes
    -----
    The function creates a list of map layers including:

    """
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

def _prepare_map_vairabibles(hideout, clickData, kdtree_distance):
    """Prepare map variables for k-d tree analysis.

    Parameters
    ----------
    hideout : dict
        Dictionary containing map settings
    clickData : dict
        Dictionary containing click data
    kdtree_distance : int
        Number of nearest neighbors to show

    Returns
    -------
    tuple
        Tuple containing:
        - stats_data: Dictionary containing statistical data
        - hideout: Dictionary containing map settings
        - colorbar: Dictionary containing colorbar settings
        - gdf: GeoDataFrame containing k-d tree data
    """
    hideout['colorscale'] = kde_colorscale
    hideout['classes'] = kde_classes
    hideout['colorProp'] = 'kde_distance'
    gdf = get_kdtree_distance(feature=clickData, gdf=stats_data_original_gdf.copy())
    gdf = gdf.sort_values(by='kde_distance').reset_index(drop=True)
    gdf = gdf.iloc[0:kdtree_distance+1]
    min_, max_ = gdf['kde_distance'].min(), gdf['kde_distance'].max()
    stats_data = gdf.__geo_interface__
    classes_colormap = np.linspace(min_, max_, num=8)
    ctg = [f"{round(cls,1)}+" for i, cls in enumerate(classes_colormap[:-1])] + [f"{round(classes_colormap[-1],1)}+"]
    colorbar = dlx.categorical_colorbar(categories= ctg,colorscale=kde_colorscale, width=500, height=30, position="bottomright")
    return stats_data, hideout, colorbar, gdf

def _prepare_map_layers_for_kmeans(stats_data, hideout):
    """Prepare map layers for k-means analysis.

    Parameters
    ----------
    stats_data : dict
        Dictionary containing statistical data
    hideout : dict
        Dictionary containing map settings

    Returns
    -------
    list
        List of map layers
    """
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
    return map_layers

def _prepare_gdf_kmeans_model(data_store_temp, kmeans_cluster, map_json):
    """Prepare GeoDataFrame and KMeans model for k-means analysis.

    Parameters
    ----------
    data_store_temp : dict
        Dictionary containing data store temporary data
    kmeans_cluster : int
        Number of clusters for k-means analysis
    map_json : dict
        Dictionary containing map JSON data

    Returns
    -------
    tuple
    """
    gdf = {}
    kmeans = {}
    if  os.path.exists(data_store_temp.get('model_stored')):
        previous_model = joblib.load(data_store_temp.get('model_stored'))
        if previous_model.n_clusters == kmeans_cluster:

            # If the number of clusters has changed, flag run the kmeans again
            gdf = gpd.GeoDataFrame.from_features(map_json['features'])
            kmeans = previous_model
    if kmeans == {}:
        _, gdf,  kmeans =  get_kmeans_cluster_add_column(kmeans_cluster, stats_data_original_gdf.copy())
    return gdf, kmeans

def _remove_model_stored_if_exists(data_store_temp, model_str):
    if os.path.exists(data_store_temp.get(model_str)):
            os.remove(data_store_temp.get(model_str))

def _generate_kmeans_scatterplot_fig(kmeans_geo_distance, selected_feature_distances_dict):
    """Generate scatter plot comparing cluster distances to geographic distances.

    Parameters
    ----------
    kmeans_geo_distance : pd.DataFrame
        DataFrame containing cluster distances and geographic distances for each area
    selected_feature_distances_dict : dict
        Dictionary containing distance metrics for the selected feature

    Returns
    -------
    plotly.graph_objects.Figure
        Scatter plot figure with:
        - Points showing cluster vs geographic distances
        - Crosshair lines highlighting selected feature
        - Polynomial regression line with R² value
        - Hover data showing city and statistical area info

    Notes
    -----
    The scatter plot shows the relationship between:
    - Distance to cluster center (x-axis)
    - Geographic distance (y-axis) 
    
    A polynomial regression line is fitted to show the trend.
    Crosshair lines highlight the selected feature's position.
    """
    # Create scatter plot attribute distance  vs geo distance
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
        # Add hover data showing coordinates
        name='Selected Area'
    )

    # Add hover data showing extra details and add customdata for further interaction
    fig_scatter.update_traces(
            hovertemplate="<br>".join([
                "Distance to Cluster: %{x:.2f}",
                "Geographic Distance: %{y:.2f} km",
                "City: %{customdata[0]}",
                "Statistical Area: %{customdata[1]}",
            ]),
            customdata=kmeans_geo_distance[['Shem_Yishuv', 'sta_22_names', 'YISHUV_STAT11']].values
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
        xshift=20,
        font=dict(size=12)
    )
    fig_scatter.update_xaxes(range=[0, x.max()])
    fig_scatter.update_yaxes(range=[0, y.max()])
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=0, b=0), title_y=0.9, title_x=0.6, font=dict(size=14), template='plotly_white',height=350, showlegend = False)
    return fig_scatter



def update_near_clster_bar(gdf, kdtree_distance, radio_map_option):
    """Update the bar plot showing nearest neighbors based on KDTree distances.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the spatial data and KDE distances
    kdtree_distance : int
        Number of nearest neighbors to show
    radio_map_option : str
        Selected analysis mode ('kdtree' or other)

    Returns
    -------
    dict
        Bar plot figure showing nearest neighbors and their distances.
        Empty dict if radio_map_option is not 'kdtree'.

    Notes
    -----
    The function:
    1. Filters out self-distances (kde_distance > 0)
    2. Takes top k nearest neighbors based on kdtree_distance
    3. Creates name labels combining settlement name and stats
    4. Generates bar plot using build_near_clsuter_bar_fig
    """
    if radio_map_option != 'kdtree':
        return {}
    else:
        # Convert GeoJSON data to GeoDataFrame
        gdf = gdf[gdf['kde_distance']>0].reset_index(drop=True)

        # Generate a barplot based on the KDE distances
        gdf_sorted = gdf.sort_values(by='kde_distance').iloc[0:kdtree_distance]

        # Concatenate settlement name and stats
        gdf_sorted['name_stat'] = gdf_sorted.apply(lambda p: p['Shem_Yishuv']+'-'+ p['sta_22_names'] if len(p['sta_22_names'])>0 else  p['Shem_Yishuv']+'-' + str(p['YISHUV_STAT11'])[-3:], axis=1) 

        # Generate the bar plot
        fig_kde = build_near_clsuter_bar_fig(gdf_sorted, kdtree_distance)
        
        return fig_kde


# Method is too long, split to subroutines and graph generators
def update_kmeans_distance_bar(gdf, feature, radio_map_option, kmeans_model):
    """Update KMeans distance bar and scatter plots based on selected feature.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the spatial data and cluster assignments
    feature : dict
        Selected feature properties including YISHUV_STAT11 ID
    radio_map_option : str
        Selected analysis mode ('kmeans' or other)
    kmeans_model : sklearn.cluster.KMeans
        Fitted KMeans model

    Returns
    -------
    tuple[dict, dict]
        fig_bar : dict
            Bar plot figure showing cluster distances
        fig_scatter : dict
            Scatter plot figure showing geographic vs cluster distances
            
    Notes
    -----
    If radio_map_option is not 'kmeans', returns empty figures.
    Otherwise:
    1. Calculates Euclidean distances to cluster centers
    2. Generates histogram of distances with vertical line for selected feature
    3. Calculates geographic distances using KDTree
    4. Creates scatter plot comparing geographic vs cluster distances
    """

    # if the radio button is not kmeans, return empty figures
    if radio_map_option != 'kmeans':
        return {}, {}
    # set a variable to store the feature id
    feature_id = -1

    # if the feature is not None, set the feature id to the feature id
    if feature is not None:
        feature_id = feature["properties"]["YISHUV_STAT11"]
    
    # prepare the dataframe for the kmeans analysis, and detect the selected feature
    df = gdf.copy()
    df = df.drop(['geometry', 'Shem_Yishuv_English',
        'Shem_Yishuv', 'Shem_Yishuv', 'sta_22_names', 'max_label'], axis=1).copy()
    
    # get the index of the selected feature
    feature_index_id = df[df['YISHUV_STAT11'] == feature_id].index[0]
    
    # drop the stats from the dataframe
    kdf_filter_row = df.loc[feature_index_id].copy().drop(['YISHUV_STAT11'])
    df.drop(['YISHUV_STAT11'], inplace=True, axis=1)
    
    # get the kmeans and the euclidean distance
    df_kmeans, eu_distance = get_kmeans_euclidian_distance(df , kdf_filter_row, kmeans_model)
    fig_bar = generate_histogram_with_line(df_kmeans, eu_distance)
    
    # in order to get the geo distance, extract the centroids of the dataframe
    gdf_centroids = gdf.copy().assign(geometry=gdf.centroid)

    # get the centroid of feature that is nearest to the cluster
    centro_filter_row = gdf_centroids.loc[df_kmeans['distance_to_cluster'].idxmin(), ['geometry', 'cluster', 'YISHUV_STAT11', 'Shem_Yishuv', 'sta_22_names']]
    gdf_filter_cluster = gdf_centroids.iloc[df_kmeans.index].copy()[['geometry', 'cluster', 'YISHUV_STAT11', 'Shem_Yishuv', 'sta_22_names' ]]

    # get the x and y coordinates of the feature that is nearest to the centroid
    centro_filter_row['x'] = centro_filter_row['geometry'].x
    centro_filter_row['y'] = centro_filter_row['geometry'].y
    centro_filter_row = centro_filter_row.drop(['cluster', 'geometry'])

    # ge the x, y of the the other features
    gdf_filter_cluster['x'] = gdf_filter_cluster['geometry'].x
    gdf_filter_cluster['y'] = gdf_filter_cluster['geometry'].y
    gdf_filter_cluster.drop(columns = ['cluster', 'geometry'], inplace=True)


    # calculate the duclidian distance between the feature and the other features using KDTree
    tree = KDTree(gdf_filter_cluster[['x', 'y']].values)

    # Query KDTree for nearest neighbors (including the row itself)
    distances, indices = tree.query(centro_filter_row[['x', 'y']].values, k=len(gdf_filter_cluster))
    gdf_filter_cluster = gdf_filter_cluster.iloc[indices].copy()
    gdf_filter_cluster['geo_distance'] = distances
    
    # sort the dataframe by the index so, it can be concantednated on the x axis
    gdf_filter_cluster = gdf_filter_cluster.sort_index()
    df_kmeans = df_kmeans.sort_index()

    # concat the geo distance and the distance to cluster (attribute distance)
    kmeans_geo_distance = pd.concat([gdf_filter_cluster[['Shem_Yishuv', 'sta_22_names' ,'YISHUV_STAT11', 'geo_distance']], df_kmeans[['distance_to_cluster']]], axis=1)
    
    # convert the geo distance to km by multiplying by 111
    kmeans_geo_distance['geo_distance'] = kmeans_geo_distance['geo_distance']*111

    # the dict will be used to generate the scatter plot
    selected_feature_distances_dict = kmeans_geo_distance.loc[feature_index_id].to_dict()
    
    fig_scatter = _generate_kmeans_scatterplot_fig(kmeans_geo_distance, selected_feature_distances_dict)
    return fig_bar, fig_scatter


### Load the data
heb_dict_df, stats_data_original_gdf = load_data_main()

#### Setting up dictionaries and classes for the map
col_rename, color_dict_party_index, color_dict_party_name = setup_col_rename_color_dicts(heb_dict_df)

# Prepare spatial data and convert Hebrew column names to English using the dictionary
stats_data_original_gdf = process_stats_data(stats_data_original_gdf, col_rename)

inital_stats_data = stats_data_original_gdf.copy().__geo_interface__

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
                html.Div([

 html.H4("\"Near things are more related than distant things\" ?"),
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
                    'display': 'flex', 'width': '100%', 'justify-content': 'space-between'})

                ],className="div-card"),
               
                html.Div(dcc.Graph(id='elections_barplot'),className="div-card"), 
                html.Div(dcc.Graph(id='kde_distance_barplot'),id='kde_distance_barplot_div', style={'display':'none'}, className="div-card"),
                html.Div([dcc.Graph(id='kmeans_distance_barplot'), dcc.Graph(id='kmeans_scatterplot')],id='kmeans_frequencybarplot_div', style={'display':'none'}, className="div-card") ], style={
                    'display': 'inline-block', 'width': '40%', 'verticalAlign': 'top',
                'minWidth': '200px'}),
            html.Div([
                dl.Map([
                    dl.TileLayer(
                        url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                    dl.LocateControl(
                        locateOptions={'enableHighAccuracy': True}),
                    dl.GeoJSON(id='stats_layer', data=inital_stats_data,
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

            ], style={'display': 'inline-block', 'width': '59%', 'verticalAlign': 'top'}, className="div-card"
            )
        ],
    ),
    dcc.Store(id='temp-data-store'),

])


@ app.callback(Output('env_map', 'children'), Output('temp-data-store', 'data'), Output('elections_barplot', 'figure'), Output('kde_distance_barplot', 'figure'), Output('kmeans_distance_barplot', 'figure'), Output('kmeans_scatterplot', 'figure'), Input('env_map', 'children'), State('stats_layer', 'data'), State('elections_barplot', 'figure'), Input('stats_layer', 'clickData'), Input('raio_map_analysis', 'value'), Input('near_cluster', 'value'), 
Input('kmeans_cluster', 'value'))
def update_map_widgets(map_layers, map_json, elections_won_fig_previous, clickData, radio_map_option, kdtree_distance, kmeans_cluster):
    """Update map widgets based on user interactions and selected analysis mode.

    Parameters
    ----------
    map_layers : list
        Current map layers configuration
    map_json : dict
        GeoJSON data for the map
    elections_won_fig_previous : dict
        Previous elections results figure
    clickData : dict
        Data from clicked location on map
    radio_map_option : str
        Selected analysis mode ('who_won', 'kdtree', or 'kmeans')
    kdtree_distance : int
        Number of nearest neighbors for KDTree analysis
    kmeans_cluster : int
        Number of clusters for KMeans analysis

    Returns
    -------
    tuple
        map_layers : list
            Updated map layers configuration
        data_store_temp : dict
            Temporary data storage including model path
        elections_won_fig : dict
            Updated elections results figure
        kde_fig : dict
            KDE distance bar plot figure (empty if not in kdtree mode)
        fig_bar : dict
            KMeans distance bar plot figure (empty if not in kmeans mode)
        fig_scatter : dict
            KMeans scatter plot figure (empty if not in kmeans mode)

    Notes
    -----
    This function handles three analysis modes:
    - 'who_won': Shows election winners
    - 'kdtree': Shows nearest neighbors using KDTree
    - 'kmeans': Shows clustering analysis using KMeans
    """
    hideout, data_store_temp, stats_data = {"color_dict":color_dict_party_index, "style":style, "hoverStyle":hover_style, 'win_party':"max_label"}, {'model_stored':'kmeans_model.joblib'}, {}
    if clickData is not None:
        stats_data = stats_data_original_gdf.copy().__geo_interface__
        elections_won_fig = generate_election_barplot_fig(clickData)
        if radio_map_option =='who_won':
            map_layers = _prepare_map_layers_for_winner(stats_data, hideout)
            _remove_model_stored_if_exists(data_store_temp, 'model_stored')
            return map_layers, data_store_temp, elections_won_fig, {}, {}, {}
       
        elif radio_map_option == 'kdtree':
            kmeans = {}
            stats_data, hideout, colorbar, gdf = _prepare_map_vairabibles(hideout, clickData, kdtree_distance)
            map_layers = _prepare_map_layers_for_kdtree(stats_data, hideout, colorbar)
            kde_fig = update_near_clster_bar(gdf, kdtree_distance, radio_map_option)
            _remove_model_stored_if_exists(data_store_temp, 'model_stored')
            return map_layers, data_store_temp, elections_won_fig,  kde_fig, {}, {}
        
        else:
            # Check if kmeans_cluster value has changed from previous run
            gdf, kmeans = _prepare_gdf_kmeans_model(data_store_temp, kmeans_cluster, map_json)

            # Get the attributes of the KMeans instance
            hideout['color_dict'], hideout['clusters_col'], stats_data = kmeans_color_dict, 'cluster', gdf.__geo_interface__
            map_layers = _prepare_map_layers_for_kmeans(stats_data, hideout)
            fig_bar, fig_scatter = update_kmeans_distance_bar(gdf, clickData, radio_map_option, kmeans)

            # Serialize the model to a byte stream
            joblib.dump(kmeans, 'kmeans_model.joblib')

            # Store the kmeans model in dcc.Store for later use
            data_store_temp = {'model_stored':'kmeans_model.joblib'}
            return map_layers, data_store_temp, elections_won_fig, {}, fig_bar, fig_scatter 
       
    if elections_won_fig_previous is None:
        elections_won_fig_previous = {}
    return map_layers, data_store_temp, elections_won_fig_previous, {}, {}, {}

@ app.callback(Output('env_map', 'viewport'), Input('kde_distance_barplot', 'clickData'), Input('kmeans_scatterplot', 'clickData'), prevent_initial_call=True)
def zoom_to_feature_by_bar(clickData1, clickData2):
    """Zoom map viewport to selected feature based on bar/scatter plot clicks.
    
    Parameters
    ----------
    clickData1 : dict or None
        Click data from KDE distance bar plot
    clickData2 : dict or None
        Click data from KMeans scatter plot
        
    Returns
    -------
    dict
        Viewport settings containing center coordinates, zoom level and transition
        animation. Empty dict if no valid click data.
    """
    stat = -1
    if clickData1 is not None:
        stat = clickData1['points'][0]['customdata'][0]
    elif clickData2 is not None:
        stat = clickData2['points'][0]['customdata'][-1]
    else:
        return {}
    centroid = stats_data_original_gdf[stats_data_original_gdf['YISHUV_STAT11'] == stat].iloc[0]['geometry'].centroid
    return dict(center=[centroid.y, centroid.x], zoom=15, transition="flyTo")


@ app.callback(Output("info", "children"), Input("stats_layer", "hoverData"))
def info_hover(feature):
    return get_info(feature = feature, col_rename=col_rename)

@ app.callback(Output("near_cluster_div", "style"), Output("kmeans_cluster_div", "style"), Output("kde_distance_barplot_div", "style"), Output("kmeans_frequencybarplot_div","style"), Input('raio_map_analysis', 'value'))
def controller(radioButton):
    """Control visibility of UI components based on selected analysis mode.

    Parameters
    ----------
    radioButton : str
        Selected analysis mode. One of:
        - 'who_won': Show election winner analysis
        - 'kdtree': Show k-d tree nearest neighbor analysis  
        - 'kmeans': Show k-means clustering analysis

    Returns
    -------
    list
        List of 4 style dictionaries controlling visibility and width of:
        - Near cluster slider div
        - K-means cluster slider div 
        - KDE distance barplot div
        - K-means frequency barplot div

    Notes
    -----
    Each style dict contains 'display' property ('none'/'block') and optionally 'width'.
    Components are shown/hidden based on the selected analysis mode.
    """
    if radioButton == 'who_won':
        return [{'display':'none'},{'display':'none'}, {'display':'none'}, {'display':'none'}]
    elif radioButton == 'kdtree':
        return [{'width': '50%', 'display':'block'}, {'display':'none'}, {'display':'block'}, {'display':'none'}]
    else:
        return [{'display':'none'}, {'width': '50%', 'display':'block'},{'display':'none'}, {'display':'block'}]

if __name__ == '__main__':
    app.run_server(debug=True)
