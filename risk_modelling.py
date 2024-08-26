import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
policy_df = pd.read_csv('Datasets/policy_data_geospatial.csv')
claims_df = pd.read_csv('Datasets/claims_data.csv')

# Sample some features for the example
# Assuming the dataset has a column 'RiskLevel' which is the target variable for classification
# and some features like 'CoverageLimit', 'Deductible', 'Premium', 'ClaimAmount', etc.

# Create a combined dataset
data = policy_df.merge(claims_df, on='PolicyID')

# Prepare the features and target variable
features = ['CoverageLimit', 'Deductible', 'Premium', 'ClaimAmount']
X = data[features]
y = data['RiskLevel']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
feature_importances = pd.DataFrame(model.feature_importances_, index=features, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
plt.title('Feature Importance')
plt.show()

# Save the model
import joblib
joblib.dump(model, 'risk_classification_model.pkl')
