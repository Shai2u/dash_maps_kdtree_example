
## Blog post (work in progress)

# Near things are more related than distant things? (Part 1)


This post is part of a series of geo-dashboard articles I wrote. If you have no/some expierence in Dash-plotly or dash-leaflet I suggest you look at: **10 Steps to Build a Geo-Dashboard with Dash-Plotly and Dash-Leaflet Maps.** Anyhow I love building geographic maps with Dash-Plotly because it allows me to quickly create and deploy somewhat complex interactive mapping user experiences and prototype my ideas in my favorite programming language, Python, while achieving React-like qualities.

In this post, I decided to take a more scientific approach and challenge the famous “First Law of Geography”: “Near things are more related than distant things” (Waldo Tobler).

In this dashboard, I chose to examine the correlation between physical geographic distance and “attribute distance.” While the results don’t provide a clear academic conclusion, they do offer some intuition for answering the question: “Are near things more related than distant things?”

In an attempt to answer this question, I wrote a two-part post. In the first part, I focus on the scientific and statistical aspects, while in the second part, I cover technical tips and tricks for Dash, Plotly, and Leaflet.

For this analysis, I focused on statistical areas and sought to identify the *n* most similar statistical areas in terms of attributes. The topic I chose to explore is election votes in Israel—not just the winner in each statistical area, but the entire voting distribution within that area.

In this way, I could determine which neighborhood(s) are most similar to mine and whether they are geographically close. However, applying a similarity index can go far beyond an election dashboard. We could analyze many other socio-economic factors or combinations of attributes, such as education, income, employment-rate, social-economic indicators, crime rates, 311 service calls, environmental qualities, or weather conditions. Additionally, examining similarity across multiple features could have practical applications, such as:

•	Apply a proven socio-economic policy from one neighborhood to others with similar attributes.
•	Examine the correlation between the lack of amenities and neighborhoods with similar characteristics.
•	Open a new business in neighborhoods similar to those where business ventures have yielded successful revenues.
•	Identify geographic/attribute patterns in areas that have experienced specific crimes or received specific 311 complaints.
•	Looking for a place to move? Examine neighborhoods that meet similar criteria to your reference area, and test if they are clustered or randomly distributed.

Let’s return to the scientific side of what I did in this dashboard — challenging the first law of geography. To do this, I applied two very different techniques:
1.	Examining attribute similarity using a K-D Tree, an efficient geographical algorithm for measuring Euclidean distance.
2.	Applying K-Means clustering to the attributes of the features, a technique that groups features around a set number of cluster centers.

I used the results of Israel’s Knesset elections and aggregated them into statistical areas. The central question I sought to answer was: Are attribute similarity distances correlated with geographic distances?


**Scientific/Statistical Techniques Explained:**

**K-D Trees**

A K-D Tree is an efficient way to compute Euclidean distances between geographic or attribute data points (think of Pythagoras’ theorem: √(x₁² + x₂²)). The K-D Tree works by partitioning points in a multidimensional space. In practice, it allows for fast distance calculations between large sets of points, making it ideal for on-the-fly computations (up to a certain limit without applying other efficiency techniques).

Using a K-D Tree, I could select a specific statistical area (or neighborhood) and treat it as the “center,” then calculate the relative distance to all other statistical areas.

For example, let’s say I live in central Tel Aviv, where 39% voted for “Yesh Atid”, 16% for “Meretz”, and 15% for “Labor”, and so on. I could then look at all other statistical areas in Israel and find those with the most similar voting patterns, regardless of geographic distance. This would allow me to identify neighborhoods in Haifa, for instance, that share similar voting attributes. Thus it's possible that central Tel Aviv and some neighborhood in Haifa might share some other attribute charactersitings while the physical distance betewen the neighborhoods is roughly 100 km.

**K-Means**

K-Means is a clustering algorithm used in machine learning and data analysis to group data points into **K** clusters. The algorithm works by iteratively finding natural cluster centers. It begins with random points, finds the closest data points to them, and then re-adjusts the center points based on the emerging clusters.

**Combining K-D Trees and K-Means**

In this exercise, I used K-Means to identify natural clusters with similar voting patterns. Since each data point is assigned a cluster index (from 0 to K), I performed the following steps:

1.	Take the row nearest to the K-Means center (for a specific cluster) and flag it as the cluster center.

2.	Compare the remaining points in the cluster to this center and calculate the Euclidean distance between them.

To efficiently calculate the Euclidean distance in N-dimensional space, I used a K-D Tree, which allows for quick distance computations on the fly.


I really enjoy exploring Dash-Plotly (and dash-leaflet) because it’s a tool that allows me to elevate dashboards to an interactive, game-like experience and achieve things that would be difficult, if not impossible, in traditional BI tools like Tableau or Power BI. For example, if I wanted to run K-Means or K-D Trees on the fly in Power BI, I’d face numerous challenges. But with Dash-Plotly, I have the full power of Python at my fingertips. In this section, I’d like to share some technical tips and explain how I overcame some challenges during the dashboard-building process.


## Multiple Views/Scenarios Based on Different Configurations
One of the powerful features that sets Dash-Plotly apart from drag-and-drop BI applications is the ability to control the UI using code. In this dashboard, I dynamically show or hide UI elements based on the selected method—whether it’s K-Means, K-D Tree, or simply displaying voting results. Another important aspect is the ability to modify both the functionality and the content displayed on the map based on user selections.


In this post, I explored K-D Trees and K-Means as methods to examine attribute Euclidean distance and tested whether it’s related to geographic distances. The value of using K-D Trees, K-Means, or a combination of both lies in the ability to test similarity across multiple dimensions (i.e., the attribute dimensions). Based on a quick exploration of Israel, it appears there isn’t a strong correlation between geography and voting trends.

In addition to the scientific analysis, I aimed to showcase how I can apply efficient Python mathematical models “on the fly” to tens of thousands of rows of data with many features.

As with my previous posts (and future ones), I love finding ways to implement Dash-Plotly together with Dash-Leaflet, challenging these technologies and discovering creative solutions to overcome their limitations. I really enjoy creating live maps, and for me, Dash-Plotly + Dash-Leaflet is a great way to practice and refine my emerging Data Science skills. It’s “reactiveness”  also inspires me on my journey to becoming a geo-driven full-stack developer.

I believe attribute similarity has great potential and practical applications. I’d love to hear your ideas in the comments. The code for this project can be found in the repository.


Caveates,
I could have also added to this assesment another test - such examining if the distribution is random or normalzied or clustered. I think I will leave for another time