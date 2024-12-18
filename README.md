# 🗺️ Dash Maps KDTree Example

This minimal Dash Plotly app uses dash-leaflet to visualize statistical areas in Israel and explore voting patterns. When a user clicks on a statistical area, the app displays other areas in Israel that are closest in terms of voting attributes, utilizing a KDTree for efficient nearest-neighbor searches. A similarity index is shown as a histogram/bar plot, which illustrates how similar the selected area is to other statistical areas, with a line chart showing a scale from 0 (closest) to 1 (furthest).

    •	Interactive map using dash-leaflet to visualize statistical areas in Israel.
    •	Nearest-neighbor search using KDTree to find areas with similar voting patterns.
    •	Similarity index visualization as a histogram/bar plot and line chart, showing the closeness of other areas to the selected one.

Requirements (to be detailed)
• Python 3.x
• Dash
• Dash-leaflet
• Plotly
• Pandas
• Scikit-learn (for KDTree)
• GeoPandas (optional, for additional geospatial data processing)

Installation
