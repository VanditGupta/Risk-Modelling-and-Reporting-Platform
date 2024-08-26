import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the policy data with geospatial information
policy_df = pd.read_csv('Datasets/policy_data_geospatial.csv')

# Create GeoDataFrame
geometry = [Point(xy)
            for xy in zip(policy_df['Longitude'], policy_df['Latitude'])]
gdf = gpd.GeoDataFrame(policy_df, geometry=geometry)

# Set CRS (Coordinate Reference System)
gdf.crs = {'init': 'epsg:4326'}

# Save the GeoDataFrame
gdf.to_file('GeoSpatial/policy_data_geospatial.gpkg', driver='GPKG')

print("Geospatial data saved successfully!")
