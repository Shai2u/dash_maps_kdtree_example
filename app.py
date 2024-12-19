from dash import Dash, dcc, html, Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
import json
import geopandas as gpd
import pandas as pd
pd.options.display.max_columns = 150
# Load the data

stats_data_gdf = gpd.read_file('data/stat_pop_simpl_votes_2022.geojson')
stats_data_gdf.to_crs('EPSG:4326', inplace=True)
stats_data = stats_data_gdf.__geo_interface__

app = Dash()
app.layout = html.Div([
    html.Div(
        [
            'hello World',
            dl.Map([
                dl.TileLayer(
                    url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'),
                dl.GeoJSON(id='stats_layer', data=stats_data,
                           options=dict(style=dict(color='blue')))
            ],
                center=[32, 34],
                zoom=7,
                style={'height': '75vh'},
                id='env_map',
                dragging=True,
                zoomControl=True,
                scrollWheelZoom=True,
                doubleClickZoom=True,
                boxZoom=True,
            )
        ],
    ),

])


if __name__ == '__main__':
    app.run_server()
