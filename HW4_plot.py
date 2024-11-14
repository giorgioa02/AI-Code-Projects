import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Data from the table
data = {
    "Destination": ["Atlanta", "Boston", "Chicago", "Dallas", "Detroit", "Denver", "Miami", 
                    "New Orleans", "New York", "Orlando", "Pittsburgh", "St. Louis"],
    "Distance": [576, 370, 612, 1216, 409, 1502, 946, 998, 189, 787, 210, 737],
    "Airfare": [178, 138, 94, 278, 158, 258, 198, 188, 98, 179, 138, 98]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Perform linear regression
X = df[['Distance']]
y = df['Airfare']
model = LinearRegression()
model.fit(X, y)

# Regression line parameters
slope = model.coef_[0]
intercept = model.intercept_

# Predict airfare for a distance of 720 miles
predicted_airfare = model.predict(pd.DataFrame([[720]], columns=['Distance']))[0]

# Print the results
print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")
print(f"Predicted Airfare for 720 miles: ${predicted_airfare:.2f}")

# Plotting
# Scatter plot of the data
plt.scatter(df['Distance'], df['Airfare'], color='blue', label='Airfare')

# Regression line
distance_range = np.linspace(df['Distance'].min(), df['Distance'].max(), 100)
predicted_airfare_line = model.predict(distance_range.reshape(-1, 1))

# Plot the regression line
plt.plot(distance_range, predicted_airfare_line, color='red', label='Linear Regression Line')

# Highlight the predicted airfare for 720 miles
plt.scatter(720, predicted_airfare, color='black', label='Estimated Airfare for 720 miles')

# Labels and title
plt.xlabel("Distance (miles)")
plt.ylabel("Airfare ($)")
plt.title("Distance vs. Airfare with Regression Line")
plt.legend()
plt.grid(True)
plt.show()