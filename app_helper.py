
from dash_extensions.javascript import arrow_function, assign
from dash import Dash, dcc, html, Input, Output


won_style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value >= classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")


kde_style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value >= classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")


style = {'color': 'white',  'fillOpacity': 0.9, 'weight': 1.5}

hover_style = {'color': 'white',  'fillOpacity': 0.5, 'weight': 3}

kde_classes = [0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84]
kde_colorscale = ['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C',
                  '#FC4E2A', '#E31A1C', '#BD0026', '#800026']

map_analysis_radio_options = [
    {'label': 'Who Won', 'value': 'who_won'},
    {'label': 'Kdtree', 'value': 'kdtree'},
    {'label': 'Kmeans', 'value': 'kmeans'}]
