import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(
    r"C:\Users\sutha\OneDrive\Documents\python code\codealpha_tasks\codealpha_sales_prediction\Advertising.csv"
)

# Check data
print(data.head())
print(data.isnull().sum())

# Features and Target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(data['TV'], data['Sales'])
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.title("Impact of Advertising Spend on Sales")
plt.show()

