"""

This script processes statistical area GeoJSON files and attaches attributes of voting trends and names of statistical areas.

Usage:
    Run this script to read GeoJSON files, process the data, and output the modified GeoJSON files with additional attributes.

Author: Shai Sussman
Date: 2024-12-18
"""

import pandas as pd
import geopandas as gpd
import os


if __name__ == "__main__":
    # File paths

    elections_project_path = '/Users/shai/Documents/Projects/Elections maps insight/elections_maps_insight'

    # Statistical areas
    gpkg_path = os.path.join(elections_project_path,
                             'data_processing/gis_data.gpkg')
    stat_layer_name = 'stat_pop_simpl_2022'

    # Statistical areas names
    stat_names = os.path.join(
        elections_project_path, '/Users/shai/Documents/Projects/Elections maps insight/elections_maps_insight/source/statistical_areas_names.xlsx')

    # Processed votes aggreagted by statistical areas
    votes_processed_path = os.path.join(
        elections_project_path, 'data_processing/stats_votes_by_statistical_stat.csv')

    # Load
    stats = gpd.read_file(gpkg_path, layer=stat_layer_name)[
        ['YISHUV_STAT11', 'Shem_Yishuv', 'Shem_Yishuv_English', 'geometry']]

    stats_name_df = pd.read_excel(stat_names)[['YISHUV_STA', 'sta_22_names']]
    stats = stats.merge(stats_name_df, how='inner',
                        left_on='YISHUV_STAT11', right_on='YISHUV_STA').drop(["YISHUV_STA"], axis=1)

    votes_df = pd.read_csv(votes_processed_path)
    stats = stats.merge(votes_df, how='inner', on='YISHUV_STAT11')

    stats.to_file(os.path.join('./data', 'stat_pop_simpl_votes_2022.geojson'),)
