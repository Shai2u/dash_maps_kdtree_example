from dash import Dash, dcc, html, Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
import json
import geopandas as gpd
import pandas as pd
pd.options.display.max_columns = 150
# Load the data

app = Dash()
app.layout = html.Div([
    html.Div(
        [
            'hello World',
            dl.Map([
                dl.TileLayer(url='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png')],
                center=[32, 34],
                zoom=7,
                style={'height': '50vh'},
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
