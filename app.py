from dash import Dash, dcc, html, Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, assign
import json

import geopandas as gpd
import pandas as pd
pd.options.display.max_columns = 150

# Load the data
heb_dict_df = pd.read_csv('data/heb_english_dict.csv', index_col=0)
col_rename = heb_dict_df.T.set_index(0)[1].to_dict()

colors_dict = heb_dict_df.T.set_index(0).loc['party_1':][2].to_dict()

classes = list(colors_dict.keys())
colorscale = list(colors_dict.values())

stats_data_gdf = gpd.read_file('data/stat_pop_simpl_votes_2022.geojson')
stats_data_gdf.to_crs('EPSG:4326', inplace=True)
stats_data_gdf.rename(columns=col_rename, inplace=True)
stats_data = stats_data_gdf.__geo_interface__


def get_info(feature=None):
    header = []
    if not feature:
        return header
    return header + [html.B(feature["properties"]["Shem_Yishuv"]), html.B(" "), html.B(feature["properties"]["sta_22_names"]), html.Br(),
                     html.Span(col_rename.get(feature["properties"]["max_label"]))]


style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value == classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")


style = {'color': 'white',  'fillOpacity': 0.5}

hover_style = {'color': 'white',  'fillOpacity': 0.9}


# Create info control.
info = html.Div(children=get_info(), id="info", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"})


app = Dash(title="Similar to me")
app.layout = html.Div([
    html.Div(
        [
            'hello World',
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
        ],
    ),

])


@app.callback(Output("info", "children"), Input("stats_layer", "hoverData"))
def info_hover(feature):
    return get_info(feature)


if __name__ == '__main__':
    app.run_server()
