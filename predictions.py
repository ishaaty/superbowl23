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
PredictX = pd.read_csv('PredictXv1.csv')
PredictXdrop = PredictX.drop('Years',1)

# algorithm
# Xdrop= Xdrop.reshape(Xdrop.shape[1:])

rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(Xdrop, ydrop)

# ML model is used to make predictions
test_preds = rf_model_on_full_data.predict(PredictXdrop)

# output = pd.DataFrame({'Dates': PredictX.Dates, 'Winner': test_preds[0][0], 'IND_Runs' : test_preds[1][1], 'NZ_Runs' : test_preds[2][2], 'IND_Wickets' : test_preds[3][3], 'NZ_Wickets' : test_preds[4][4]})
output = pd.DataFrame({'Years': PredictX.Years, 'Color' : test_preds})

print(output)

output.to_csv('Predictions.csv', index=False)

# yellow if eagles win
# orange if chiefs win