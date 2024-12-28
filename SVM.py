import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Scale the features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVR model
svr_model = SVR(kernel='rbf')  # You can also try 'linear' or 'poly' kernels

# Fit the model to the training data
svr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svr_model.predict(X_test)

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
plt.title("Actual vs. Predicted Population Density (SVR)")

# Add a diagonal line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

# Show the plot
plt.show()
