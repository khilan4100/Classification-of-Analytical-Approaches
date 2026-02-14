import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("sales_data.csv")
print(data.head())

print(data.info())

print(data.describe())

total_sales = data['Sales'].sum()
average_sales = data['Sales'].mean()

print("\nTotal Sales:", total_sales)
print("Average Sales:", average_sales)

monthly_sales = data.groupby('Month')['Sales'].sum()
monthly_sales.plot(kind='bar')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

region_sales = data.groupby('Region')['Sales'].mean()
print(region_sales)

correlation = data.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(correlation)

sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()

low_profit = data[data['Profit'] < data['Profit'].mean()]
print("\nLow Profit Months:")
print(low_profit)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = data[['Advertising_Spend']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, predictions))

future_sales = model.predict([[6000]])
print("Predicted Sales for ₹6000 Advertising:", future_sales)

if future_sales > average_sales:
    print("\nRecommendation: Increase Advertising Budget")
else:
    print("\nRecommendation: Optimize Cost Strategy")

