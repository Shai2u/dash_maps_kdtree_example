# 🗺️ Near things are more related than distant things?

![Screenshot](./media/screenshot.png)

Geo-Dashboard: Examining Attribute and Geographic Similarities

This project explores the correlation between geographic proximity and attribute similarity using K-D Trees and K-Means clustering, focusing on voting trends in Israel. It’s part of a series of articles about building geo-dashboards using Dash-Plotly and Dash-Leaflet. I wanted to challenge the famous “First Law of Geography” by examining if near things are more related than distant things, in terms of voting patterns across different statistical areas.

Key Features:
	•	Attribute Similarity Analysis: Using K-D Trees and K-Means clustering, this dashboard explores the Euclidean distance between statistical areas based on their voting patterns.
	•	Scientific Exploration: Investigating the correlation between geographic distance and attribute similarity (such as election votes) to understand if proximity leads to similar socio-economic or political patterns.
	•	Dynamic Dashboard: The dashboard allows users to interactively explore the geographical and attribute-based similarity of neighborhoods, and tests how similar statistical areas can be to one another, irrespective of geographic distance.
	•	Interactive Live Maps: Built with Dash-Plotly and Dash-Leaflet, this project creates a live interactive experience to explore the data visually.

Potential Applications:
	•	Apply socio-economic policies to similar neighborhoods.
	•	Examine the relationship between neighborhood characteristics and service availability.
	•	Identify patterns in crime rates, service requests (311 complaints), and more.
	•	Find areas with similar attributes for business expansion or relocation.

Techniques Used:
	•	K-D Trees: Used for efficient calculation of Euclidean distances between neighborhoods based on various attributes.
	•	K-Means Clustering: Used to group neighborhoods based on similar voting patterns.
	•	Dash-Plotly & Dash-Leaflet: Utilized to create dynamic, interactive maps for data exploration.

Requirements (to be detailed)

- Python 3.x
- Dash
- Dash-leaflet
- Plotly
- Pandas
- Scikit-learn (for KDTree)
- GeoPandas (optional, for additional geospatial data processing)

Installation (to be detailed)

