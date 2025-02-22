
from dash_extensions.javascript import arrow_function, assign
from dash import Dash, dcc, html, Input, Output


won_style_handle = assign("""function(feature, context){
    const {color_dict, style, win_party} = context.hideout;  // get props from hideout
    const value = feature.properties[win_party];  // get value the determines the color
    style.fillColor = color_dict[value];  // set the fill color according to the class

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

kmeans_style_handle = assign("""function(feature, context){
    const {color_dict, style, clusters_col} = context.hideout;  // get props from hideout
    const value = feature.properties[clusters_col];  // get value the determines the color
    style.fillColor = color_dict[value];  // set the fill color according to the class

    return style;
}""")

style = {'color': 'white',  'fillOpacity': 0.9, 'weight': 1.5}

hover_style = {'color': 'white',  'fillOpacity': 0.5, 'weight': 3}

kde_classes = [0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84]
kde_colorscale = ['#112A32', '#118443', '#41AB5D', '#78C679', '#ADDD8E', '#D9F0A3', '#F7FCB9', '#FFFFE5',]
kmeans_color_dict = {0: '#FF5733',
                    1: '#33FF57',
                    2: '#3357FF',
                    3: '#FF33A1',
                    4: '#A133FF',
                    5: '#33FFA1',
                    6: '#FF8C33',
                    7: '#33FFF5',
                    8: '#F533FF',
                    9: '#39FFAA'}
map_analysis_radio_options = [
    {'label': 'Who Won', 'value': 'who_won'},
    {'label': 'KD-Tree', 'value': 'kdtree'},
    {'label': 'K-Means', 'value': 'kmeans'}]
