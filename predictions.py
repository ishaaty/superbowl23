# imports libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# reads & describes data from files
X = pd.read_csv('Xdata.csv')
y = pd.read_csv('Ydata.csv')
print(X.describe())

# drops the column name 'Date' from the dataset
Xdrop = X.drop('Years',1)
ydrop = y.drop('Years',1)

# reads data used for predictions
PredictX = pd.read_csv('PredictXv2.csv')
PredictXdrop = PredictX.drop('Years',1)

# algorithm
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(Xdrop, ydrop)

# ML model is used to make predictions
test_preds = rf_model_on_full_data.predict(PredictXdrop)

output = pd.DataFrame({'Years': PredictX.Years, 'Color' : test_preds})

print(output)

output.to_csv('Predictions.csv', index=False)

# yellow if eagles win
# orange if chiefs win