

## Blog post (work in progress)

# Near things are more related than distant things? (Part 2)


This post is part of a series of geo-dashboard articles I wrote. If you have no expierence in Dash-plotly or dash-leaflet I suggest you look at: **10 Steps to Build a Geo-Dashboard with Dash-Plotly and Dash-Leaflet Maps.** Anyhow I love building geographic maps with Dash-Plotly because it allows me to quickly create and deploy somewhat complex interactive mapping user experiences and prototype my ideas in my favorite programming language, Python, while achieving React-like qualities.

In this post, I decided to take a more scientific approach and challenge the famous “First Law of Geography”: “Near things are more related than distant things” (Waldo Tobler).

In this dashboard, I chose to examine the correlation between physical geographic distance and “attribute distance.” While the results don’t provide a clear academic conclusion, they do offer some intuition for answering the question: “Are near things more related than distant things?”

In an attempt to answer this question, I wrote a two-part post. In the first part, I focus on the scientific and statistical aspects, while in the second part, I cover technical tips and tricks for Dash, Plotly, and Leaflet.

If you haven't read this first part ()[] if you choose to skip the read here is the TL;DR version.
(Add apart about the dashboard)


Some tips for tackeling multiple views and scenarios in the dashbaord:
1. 	To control the views (what is displayed and what is hidden), I borrowed a concept from JavaScript—toggling the visibility of Div elements. This is done by adding an output callback to one of the Div elements and modifying its style attribute.

Python'''
@ app.callback(Output("near_cluster_div", "style"), Output("kmeans_cluster_div", "style"), Output("kde_distance_barplot_div", "style"), Output("kmeans_frequencybarplot_div","style"), Input('raio_map_analysis', 'value'))
def controller(radioButton):
    if radioButton == 'who_won':
        return [{'display':'none'},{'display':'none'}, {'display':'none'}, {'display':'none'}]
    elif radioButton == 'kdtree':
        return [{'width': '50%', 'display':'block'}, {'display':'none'}, {'display':'block'}, {'display':'none'}]
    else:
        return [{'display':'none'}, {'width': '50%', 'display':'block'},{'display':'none'}, {'display':'block'}]

'''
2. Centrelize callbacks as much as possible
I use a single callback to generate all figures for every app scenario instead of separate callbacks to reduce redundancy and maintain a structured data flow. This prevents parallel execution issues, avoiding inconsistencies and redundant data processing when the app configuration changes.

3. Return empty figures to avoid processing
In Plotly Dash, returning empty figures (e.g., go.Figure() or {}) for inactive components helps avoid unnecessary computations and improves performance. This ensures that only relevant graphs are processed, reducing load time and preventing errors from rendering unnecessary data.

4. Use the map layers as an input for data
In Plotly Dash, it’s better to store data within components (such as dcc.Store) rather than using global variables. This ensures that the app remains stateless, avoiding issues with concurrent users and unintended data modifications. By storing the map layer data as JSON inside a the map-layers with dash-leaflet, you can safely retrieve, analyze, and update it without affecting other users or interfering with the app’s global state.


gdf = gpd.GeoDataFrame.from_features(map_json['features'])

- Multiple Scenarios in the app.
    - Hiding and showing the right div - returning empty divs
    - Working with multiple map-layers configurations based on the scenario
- Avoiding extensive runs by examining the state of the model.
    - Sateralizing??? a model
- working with two way callbacks, where the map controls the graph and the graph controls the map
- Using the Map layer is the source of data

Dash apps tend to become very long and messy, often resembling spaghetti code. To keep your code clean and maintainable, consider the following:
	1.	Refactor, Refactor, Refactor! Continuously improve and restructure your code.
	2.	Step Away and Reevaluate – If your code becomes too long and difficult to follow, take a break for a few days. When you return, review it with a bird’s-eye view and logically restructure it.
	3.	Use External Files for Configuration and Reusable Methods – If you have configuration variables or general-purpose methods that aren’t directly tied to your app’s logic, move them to separate files. Keep your functions short, clear, and well-structured, avoiding deep hierarchical complexity.
    4. .	Don’t hesitate to use GPT agents to restructure your code, add comments, write docstrings, and format your document properly. It’s a real time-saver!