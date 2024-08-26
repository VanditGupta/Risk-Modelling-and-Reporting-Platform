import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

# Load data
policy_df = pd.read_csv('Datasets/policy_data_geospatial.csv')
claims_df = pd.read_csv('Datasets/claims_data.csv')
predictions_df = pd.read_csv('Datasets/Results/predicted_claims.csv')

# Sample a subset of the policy data for better visualization (e.g., 10% of the data)
policy_sample_df = policy_df.sample(frac=0.1, random_state=42)

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(
    policy_sample_df['Longitude'], policy_sample_df['Latitude'])]
gdf = gpd.GeoDataFrame(policy_sample_df, geometry=geometry)

# Create figures
fig_policies = px.histogram(
    policy_df, x='CoverageLimit', title='Policy Coverage Limits')
fig_claims = px.histogram(claims_df, x='ClaimAmount', title='Claim Amounts')
fig_predictions = px.scatter(
    predictions_df, x='Actual', y='Predicted', title='Actual vs Predicted Claims')

# Geospatial plot
fig_geospatial = px.scatter_mapbox(policy_sample_df, lat='Latitude', lon='Longitude', hover_name='PolicyID',
                                   hover_data=['CoverageLimit', 'Premium'],
                                   color_discrete_sequence=["fuchsia"], zoom=1, height=600)
fig_geospatial.update_layout(mapbox_style="open-street-map")
fig_geospatial.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Risk Modelling and Reporting Dashboard"),
    dcc.Graph(id='coverage-limits', figure=fig_policies),
    dcc.Graph(id='claim-amounts', figure=fig_claims),
    dcc.Graph(id='predicted-claims', figure=fig_predictions),
    dcc.Graph(id='geospatial-distribution', figure=fig_geospatial)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
