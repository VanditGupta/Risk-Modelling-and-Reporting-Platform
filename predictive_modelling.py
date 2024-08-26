import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
policy_df = pd.read_csv('Datasets/policy_data_geospatial.csv')
claims_df = pd.read_csv('Datasets/claims_data.csv')

# Prepare data
X = policy_df[['CoverageLimit', 'Deductible', 'Premium']]
y = claims_df['ClaimAmount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('Datasets/Results/predicted_claims.csv', index=False)

print("Claims prediction completed successfully!")
