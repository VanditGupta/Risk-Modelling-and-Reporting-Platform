import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Load the GeoDataFrame
gdf = gpd.read_file('GeoSpatial/policy_data_geospatial.gpkg')

# Sample a subset of the data for better visualization (e.g., 1% of the data)
gdf_sample = gdf.sample(frac=0.01, random_state=42)

# Create a heatmap of the points
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world = gpd.read_file(
    'ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
world.boundary.plot(ax=ax, linewidth=1)

# Create a 2D histogram of the data points
h, xedges, yedges = np.histogram2d(
    gdf_sample['Longitude'], gdf_sample['Latitude'], bins=100)

# Plot the heatmap
pcm = ax.imshow(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower', cmap='hot', norm=LogNorm(), alpha=0.6)
fig.colorbar(pcm, ax=ax, label='Count (log scale)')
plt.title('Heatmap of Insurance Policies (Sampled)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
