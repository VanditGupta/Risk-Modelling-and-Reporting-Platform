import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
num_rows = 50000

# Generating data
policy_data = {
    'PolicyID': np.arange(1, num_rows + 1),
    'PolicyType': np.random.choice(['Homeowners', 'Auto', 'Commercial'], num_rows),
    'CoverageLimit': np.round(np.random.uniform(50000, 1000000, num_rows), 2),
    'Deductible': np.round(np.random.uniform(500, 10000, num_rows), 2),
    'Premium': np.round(np.random.uniform(200, 10000, num_rows), 2),
    'StartDate': [datetime(2010, 1, 1) + timedelta(days=random.randint(0, 3650)) for _ in range(num_rows)],
    'EndDate': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 3650)) for _ in range(num_rows)],
    'Latitude': np.round(np.random.uniform(-90, 90, num_rows), 6),
    'Longitude': np.round(np.random.uniform(-180, 180, num_rows), 6)
}

claims_data = {
    'ClaimID': np.arange(1, num_rows + 1),
    'PolicyID': np.random.choice(policy_data['PolicyID'], num_rows),
    'ClaimAmount': np.round(np.random.uniform(1000, 50000, num_rows), 2),
    'DateOfLoss': [datetime(2010, 1, 1) + timedelta(days=random.randint(0, 3650)) for _ in range(num_rows)],
    'CauseOfLoss': np.random.choice(['Fire', 'Collision', 'Theft', 'Natural Disaster'], num_rows),
    'SettlementDate': [datetime(2010, 1, 1) + timedelta(days=random.randint(0, 3650)) for _ in range(num_rows)],
    'Status': np.random.choice(['Settled', 'Pending', 'Rejected'], num_rows)
}

investment_data = {
    'InvestmentID': np.arange(1, num_rows + 1),
    'AssetType': np.random.choice(['Bond', 'Stock', 'Real Estate'], num_rows),
    'AssetValue': np.round(np.random.uniform(10000, 1000000, num_rows), 2),
    'PurchaseDate': [datetime(2000, 1, 1) + timedelta(days=random.randint(0, 7300)) for _ in range(num_rows)],
    'CurrentValue': np.round(np.random.uniform(10000, 1000000, num_rows), 2),
    'MaturityDate': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 7300)) for _ in range(num_rows)]
}

# Create DataFrames
policy_df = pd.DataFrame(policy_data)
claims_df = pd.DataFrame(claims_data)
investment_df = pd.DataFrame(investment_data)

# Save to CSV files
policy_df.to_csv('Datasets/policy_data_geospatial.csv', index=False)
claims_df.to_csv('Datasets/claims_data.csv', index=False)
investment_df.to_csv('Datasets/investment_data.csv', index=False)

print("Data generated successfully!")
