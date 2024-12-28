import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("processed_train_data.csv") 

# Drop rows with missing values
df = df.dropna()

# Separate features (X) and target variable (y)
X = df.drop('population_density', axis=1) 
y = df['population_density']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42) 

# Fit the model to the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Create a scatter plot
plt.scatter(y_test, y_pred)

# Add labels and title
plt.xlabel("Actual Population Density")
plt.ylabel("Predicted Population Density")
plt.title("Actual vs. Predicted Population Density (Random Forest Regression)")

# Add a diagonal line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

# Show the plot
plt.show()